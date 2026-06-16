#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <pipewire/pipewire.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <spa/param/audio/format-utils.h>

#include "onnxruntime_c_api.h"

#define MODEL_RATE 16000U
#define MODEL_SECONDS 3U
#define MODEL_SAMPLES (MODEL_RATE * MODEL_SECONDS)
#define RING_SECONDS 4U
#define DEFAULT_THRESHOLD 0.6973556280136108f
#define DEFAULT_CONSECUTIVE 2U
#define CAPTURE_HEALTH_TIMEOUT_SECONDS 10.0
#define WORKER_MAX_SLEEP_SECONDS 0.5
#define WORKER_MIN_SLEEP_SECONDS 0.001
#define CAPTURE_STATUS_FILENAME "native-listener.capture-status"

struct sample_ring {
    /*
     * The ring is the short looping tape.
     *
     * The PipeWire callback continuously records model-rate audio onto this
     * tape: 16 kHz, mono, float32. When write_index reaches the end, it wraps
     * and overwrites the oldest samples. We keep 4 seconds so the worker can
     * always read a complete 3-second wakeword window without asking PipeWire
     * for audio again.
     */
    float *samples;
    size_t capacity;
    _Atomic size_t write_index;
    _Atomic uint64_t total_written;
};

struct listener_options {
    const char *model_path;
    const char *handler_command;
    const char *activation_config_path;
    const char *hermes_home;
    const char *target_object;
    float threshold;
    unsigned int consecutive;
    unsigned int duration_seconds;
    unsigned int inference_interval_ms;
    bool verbose;
    bool self_test;
    char derived_hermes_home[4096];
    char derived_model_path[4096];
    char derived_activation_config_path[4096];
};

struct wake_model {
    const OrtApi *ort;
    OrtEnv *env;
    OrtSessionOptions *session_options;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    OrtAllocator *allocator;
    char *input_name;
    char *output_name;
};

struct listener_data {
    struct pw_main_loop *loop;
    struct pw_stream *stream;
    struct spa_audio_info format;
    struct sample_ring ring;
    struct wake_model model;
    pthread_t worker;
    _Atomic bool running;
    _Atomic bool worker_started;
    _Atomic unsigned int negotiated_rate;
    _Atomic unsigned int negotiated_channels;
    unsigned int resample_accumulator;
    char capture_status_path[4096];
    struct listener_options options;
};

static void ring_destroy(struct sample_ring *ring)
{
    free(ring->samples);
    ring->samples = NULL;
    ring->capacity = 0;
}

static int ring_init(struct sample_ring *ring, size_t capacity)
{
    ring->samples = calloc(capacity, sizeof(float));
    if (ring->samples == NULL)
        return -errno;
    ring->capacity = capacity;
    atomic_store(&ring->write_index, 0);
    atomic_store(&ring->total_written, 0);
    return 0;
}

static void ring_write(struct sample_ring *ring, const float *samples, size_t n_samples)
{
    size_t start = atomic_load_explicit(&ring->write_index, memory_order_relaxed);

    if (ring->capacity == 0 || samples == NULL || n_samples == 0)
        return;

    for (size_t i = 0; i < n_samples; i++)
        ring->samples[(start + i) % ring->capacity] = samples[i];

    atomic_store_explicit(&ring->write_index, (start + n_samples) % ring->capacity, memory_order_release);
    atomic_fetch_add_explicit(&ring->total_written, n_samples, memory_order_release);
}

static size_t ring_snapshot_latest(struct sample_ring *ring, float *out, size_t n_samples)
{
    uint64_t total = atomic_load_explicit(&ring->total_written, memory_order_acquire);
    size_t write_index = atomic_load_explicit(&ring->write_index, memory_order_acquire);
    size_t available = total < ring->capacity ? (size_t)total : ring->capacity;
    size_t count = n_samples < available ? n_samples : available;
    size_t start;

    if (count == 0 || out == NULL)
        return 0;

    /*
     * Copy the latest stretch off the tape into a plain linear buffer. ONNX
     * wants one contiguous tensor; the latest audio in the ring may wrap
     * across the end/start boundary, so the worker uses this snapshot as the
     * model input for one inference run.
     */
    start = (write_index + ring->capacity - count) % ring->capacity;
    for (size_t i = 0; i < count; i++)
        out[i] = ring->samples[(start + i) % ring->capacity];

    return count;
}

static double monotonic_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void sleep_worker_tick(void)
{
    struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = WORKER_POLL_NS,
    };

    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
    }
}

static int join_hermes_path(char *out, size_t out_size, const char *hermes_home, const char *relative_path)
{
    int written;

    if (hermes_home == NULL || hermes_home[0] == '\0')
        return -1;
    written = snprintf(out, out_size, "%s/%s", hermes_home, relative_path);
    if (written < 0 || (size_t)written >= out_size)
        return -1;
    return 0;
}

static int resolve_hermes_paths(struct listener_options *options)
{
    const char *env_hermes_home;
    const char *home;
    int written;

    if (options->hermes_home == NULL || options->hermes_home[0] == '\0') {
        env_hermes_home = getenv("HERMES_HOME");
        if (env_hermes_home != NULL && env_hermes_home[0] != '\0') {
            options->hermes_home = env_hermes_home;
        } else {
            home = getenv("HOME");
            if (home == NULL || home[0] == '\0')
                return -1;
            written = snprintf(options->derived_hermes_home,
                               sizeof(options->derived_hermes_home),
                               "%s/.hermes",
                               home);
            if (written < 0 || (size_t)written >= sizeof(options->derived_hermes_home))
                return -1;
            options->hermes_home = options->derived_hermes_home;
        }
    }

    if (options->model_path == NULL || options->model_path[0] == '\0') {
        if (join_hermes_path(options->derived_model_path,
                             sizeof(options->derived_model_path),
                             options->hermes_home,
                             "wakeword/okay-hermes-repcnn-onnx/wakeword.fixed-1x48000.onnx") < 0)
            return -1;
        options->model_path = options->derived_model_path;
    }

    if (options->activation_config_path == NULL || options->activation_config_path[0] == '\0') {
        if (join_hermes_path(options->derived_activation_config_path,
                             sizeof(options->derived_activation_config_path),
                             options->hermes_home,
                             "wakeword/config.yaml") < 0)
            return -1;
        options->activation_config_path = options->derived_activation_config_path;
    }

    return 0;
}

static void resolve_capture_status_path(struct listener_data *data)
{
    if (join_hermes_path(data->capture_status_path,
                         sizeof(data->capture_status_path),
                         data->options.hermes_home,
                         "wakeword/" CAPTURE_STATUS_FILENAME) < 0)
        data->capture_status_path[0] = '\0';
}

static void write_capture_status(const struct listener_data *data, const char *status)
{
    char tmp_path[4352];
    FILE *file;
    int written;

    if (data->capture_status_path[0] == '\0')
        return;
    written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", data->capture_status_path, (long)getpid());
    if (written < 0 || (size_t)written >= sizeof(tmp_path))
        return;

    file = fopen(tmp_path, "w");
    if (file == NULL)
        return;
    fprintf(file, "%s\n", status);
    if (fclose(file) != 0) {
        unlink(tmp_path);
        return;
    }
    if (rename(tmp_path, data->capture_status_path) != 0)
        unlink(tmp_path);
}

static void clear_capture_status(const struct listener_data *data)
{
    if (data->capture_status_path[0] != '\0')
        unlink(data->capture_status_path);
}

static int ort_status_ok(struct wake_model *model, OrtStatus *status, const char *context)
{
    if (status == NULL)
        return 0;
    fprintf(stderr, "%s: %s\n", context, model->ort->GetErrorMessage(status));
    model->ort->ReleaseStatus(status);
    return -1;
}

static int wake_model_init(struct wake_model *model, const char *model_path)
{
    memset(model, 0, sizeof(*model));
    model->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (model->ort == NULL) {
        fprintf(stderr, "could not resolve ONNX Runtime C API\n");
        return -1;
    }

    if (ort_status_ok(model, model->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "okay-hermes-wake-listener", &model->env), "CreateEnv") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->CreateSessionOptions(&model->session_options), "CreateSessionOptions") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->SetIntraOpNumThreads(model->session_options, 1), "SetIntraOpNumThreads") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->SetInterOpNumThreads(model->session_options, 1), "SetInterOpNumThreads") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->SetSessionExecutionMode(model->session_options, ORT_SEQUENTIAL), "SetSessionExecutionMode") < 0)
        return -1;
    /*
     * The wake model is tiny and fixed-shape. ONNX Runtime's CPU arena and
     * memory-pattern planner are useful for larger/dynamic workloads, but here
     * they mostly hold extra resident heap after session creation.
     */
    if (ort_status_ok(model, model->ort->DisableCpuMemArena(model->session_options), "DisableCpuMemArena") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->DisableMemPattern(model->session_options), "DisableMemPattern") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->CreateSession(model->env, model_path, model->session_options, &model->session), "CreateSession") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->GetAllocatorWithDefaultOptions(&model->allocator), "GetAllocatorWithDefaultOptions") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->SessionGetInputName(model->session, 0, model->allocator, &model->input_name), "SessionGetInputName") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->SessionGetOutputName(model->session, 0, model->allocator, &model->output_name), "SessionGetOutputName") < 0)
        return -1;
    if (ort_status_ok(model, model->ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &model->memory_info), "CreateCpuMemoryInfo") < 0)
        return -1;
    return 0;
}

static void wake_model_destroy(struct wake_model *model)
{
    if (model->allocator != NULL && model->input_name != NULL)
        model->allocator->Free(model->allocator, model->input_name);
    if (model->allocator != NULL && model->output_name != NULL)
        model->allocator->Free(model->allocator, model->output_name);
    if (model->memory_info != NULL)
        model->ort->ReleaseMemoryInfo(model->memory_info);
    if (model->session != NULL)
        model->ort->ReleaseSession(model->session);
    if (model->session_options != NULL)
        model->ort->ReleaseSessionOptions(model->session_options);
    if (model->env != NULL)
        model->ort->ReleaseEnv(model->env);
}

static int run_model(struct wake_model *model, float *input, float *score)
{
    int64_t shape[2] = {1, MODEL_SAMPLES};
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    const char *input_names[1] = {model->input_name};
    const char *output_names[1] = {model->output_name};
    float *output_data = NULL;

    if (ort_status_ok(model,
                      model->ort->CreateTensorWithDataAsOrtValue(model->memory_info,
                                                                 input,
                                                                 MODEL_SAMPLES * sizeof(float),
                                                                 shape,
                                                                 2,
                                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                 &input_tensor),
                      "CreateTensorWithDataAsOrtValue") < 0)
        return -1;

    if (ort_status_ok(model,
                      model->ort->Run(model->session,
                                      NULL,
                                      input_names,
                                      (const OrtValue *const *)&input_tensor,
                                      1,
                                      output_names,
                                      1,
                                      &output_tensor),
                      "OrtRun") < 0) {
        model->ort->ReleaseValue(input_tensor);
        return -1;
    }

    if (ort_status_ok(model, model->ort->GetTensorMutableData(output_tensor, (void **)&output_data), "GetTensorMutableData") < 0) {
        model->ort->ReleaseValue(output_tensor);
        model->ort->ReleaseValue(input_tensor);
        return -1;
    }

    *score = output_data[0];
    model->ort->ReleaseValue(output_tensor);
    model->ort->ReleaseValue(input_tensor);
    return 0;
}

static int run_handler_command(const char *command,
                               const char *hermes_home,
                               const char *activation_config_path,
                               float probability,
                               unsigned int sample_rate,
                               unsigned int channels)
{
    int pipefd[2];
    pid_t pid;
    char payload[512];
    char default_command[4096];
    ssize_t written;
    int status = 0;
    struct timespec now;

    if ((command == NULL || command[0] == '\0') && activation_config_path != NULL && activation_config_path[0] != '\0') {
        snprintf(default_command,
                 sizeof(default_command),
                 "%s/hermes-agent/venv/bin/python -m okay_hermes_voice.native_activation_handler --config %s",
                 hermes_home,
                 activation_config_path);
        command = default_command;
    }

    if (command == NULL || command[0] == '\0') {
        printf("{\"event\":\"activation\",\"probability\":%.6f,\"sample_rate\":%u,\"channels\":%u}\n",
               probability,
               sample_rate,
               channels);
        fflush(stdout);
        return 0;
    }

    clock_gettime(CLOCK_REALTIME, &now);
    snprintf(payload,
             sizeof(payload),
             "{\"probability\":%.9f,\"scores\":[%.9f],\"sample_rate\":%u,\"detected_at\":%.9f,\"native_listener\":true}\n",
             probability,
             probability,
             MODEL_RATE,
             (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0);

    if (pipe(pipefd) < 0)
        return -1;

    pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        _exit(127);
    }

    close(pipefd[0]);
    written = write(pipefd[1], payload, strlen(payload));
    (void)written;
    close(pipefd[1]);
    if (waitpid(pid, &status, 0) < 0)
        return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void *run_wakeword_worker(void *userdata)
{
    struct listener_data *data = userdata;
    /*
     * The worker model buffer is not another recorder. It is the read head's
     * scratch pad: every inference interval, copy the newest 3 seconds from
     * the looping tape into this contiguous tensor-shaped buffer, then hand it
     * to ONNX Runtime. The callback keeps recording while this happens.
     */
    float *model_input = calloc(MODEL_SAMPLES, sizeof(float));
    uint64_t last_total = 0;
    unsigned int hits = 0;
    double last_inference = 0.0;
    double inference_interval = (double)data->options.inference_interval_ms / 1000.0;

    if (model_input == NULL) {
        fprintf(stderr, "failed to allocate wakeword worker buffer\n");
        atomic_store(&data->running, false);
        pw_main_loop_quit(data->loop);
        return NULL;
    }

    while (atomic_load_explicit(&data->running, memory_order_acquire)) {
        uint64_t total = atomic_load_explicit(&data->ring.total_written, memory_order_acquire);
        size_t copied;
        float probability = 0.0f;

        if (total != last_total && total >= MODEL_SAMPLES) {
            double now = monotonic_seconds();
            if (now - last_inference < inference_interval) {
                sleep_worker_tick();
                continue;
            }
            last_inference = now;
            copied = ring_snapshot_latest(&data->ring, model_input, MODEL_SAMPLES);
            if (copied != MODEL_SAMPLES) {
                sleep_worker_tick();
                continue;
            }
            if (run_model(&data->model, model_input, &probability) == 0) {
                if (data->options.verbose)
                    fprintf(stderr, "wake-listener score=%.6f rate=%u channels=1\n", probability, MODEL_RATE);
                if (probability >= data->options.threshold)
                    hits++;
                else
                    hits = 0;
                if (hits >= data->options.consecutive) {
                    run_handler_command(data->options.handler_command,
                                        data->options.hermes_home,
                                        data->options.activation_config_path,
                                        probability,
                                        MODEL_RATE,
                                        1);
                    hits = 0;
                }
            }
            last_total = total;
        }
        sleep_worker_tick();
    }

    free(model_input);
    return NULL;
}

static void on_process(void *userdata)
{
    struct listener_data *data = userdata;
    struct pw_buffer *buffer = pw_stream_dequeue_buffer(data->stream);
    struct spa_buffer *spa_buffer;
    struct spa_data *spa_data;
    struct spa_chunk *chunk;
    const uint8_t *bytes;
    const float *samples;
    size_t n_samples;
    unsigned int rate;
    unsigned int channels;
    size_t frames;

    if (buffer == NULL)
        return;

    spa_buffer = buffer->buffer;
    spa_data = &spa_buffer->datas[0];
    chunk = spa_data->chunk;

    if (spa_data->data != NULL && chunk != NULL && chunk->size > 0 &&
        data->format.info.raw.format == SPA_AUDIO_FORMAT_F32) {
        bytes = (const uint8_t *)spa_data->data + chunk->offset;
        samples = (const float *)bytes;
        n_samples = chunk->size / sizeof(float);
        rate = atomic_load_explicit(&data->negotiated_rate, memory_order_relaxed);
        channels = atomic_load_explicit(&data->negotiated_channels, memory_order_relaxed);
        if (rate == 0)
            rate = MODEL_RATE;
        if (channels == 0)
            channels = 1;
        frames = n_samples / channels;

        /*
         * Record onto the tape in the model's native shape, not PipeWire's
         * negotiated shape. The mic may arrive as 48 kHz stereo; the wakeword
         * model only consumes 16 kHz mono, so storing anything larger just
         * burns RAM. This loop is intentionally simple and bounded: downmix one
         * frame, advance a fixed-point-ish accumulator, and write only the
         * output samples that belong on the 16 kHz tape.
         */
        for (size_t frame = 0; frame < frames; frame++) {
            float mono;
            double sum = 0.0;

            for (unsigned int ch = 0; ch < channels; ch++)
                sum += samples[frame * channels + ch];
            mono = (float)(sum / (double)channels);

            data->resample_accumulator += MODEL_RATE;
            while (data->resample_accumulator >= rate) {
                ring_write(&data->ring, &mono, 1);
                data->resample_accumulator -= rate;
            }
        }
    }

    pw_stream_queue_buffer(data->stream, buffer);
}

static void on_stream_param_changed(void *userdata, uint32_t id, const struct spa_pod *param)
{
    struct listener_data *data = userdata;

    if (param == NULL || id != SPA_PARAM_Format)
        return;

    if (spa_format_parse(param, &data->format.media_type, &data->format.media_subtype) < 0)
        return;

    if (data->format.media_type != SPA_MEDIA_TYPE_audio ||
        data->format.media_subtype != SPA_MEDIA_SUBTYPE_raw)
        return;

    spa_format_audio_raw_parse(param, &data->format.info.raw);
    data->resample_accumulator = 0;
    atomic_store(&data->negotiated_rate, data->format.info.raw.rate);
    atomic_store(&data->negotiated_channels, data->format.info.raw.channels);
    fprintf(stderr,
            "capturing rate:%u channels:%u format:%d\n",
            data->format.info.raw.rate,
            data->format.info.raw.channels,
            data->format.info.raw.format);
}

static void on_state_changed(void *userdata,
                             enum pw_stream_state old,
                             enum pw_stream_state state,
                             const char *error)
{
    struct listener_data *data = userdata;
    (void)old;

    if (state == PW_STREAM_STATE_ERROR) {
        fprintf(stderr, "PipeWire stream error: %s\n", error == NULL ? "unknown" : error);
        atomic_store(&data->running, false);
        pw_main_loop_quit(data->loop);
    }
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_state_changed,
    .param_changed = on_stream_param_changed,
    .process = on_process,
};

static void quit_signal(void *userdata, int signal_number)
{
    struct listener_data *data = userdata;
    (void)signal_number;

    atomic_store(&data->running, false);
    pw_main_loop_quit(data->loop);
}

static void *duration_thread(void *userdata)
{
    struct listener_data *data = userdata;

    for (unsigned int i = 0; i < data->options.duration_seconds; i++) {
        if (!atomic_load_explicit(&data->running, memory_order_acquire))
            return NULL;
        sleep(1);
    }

    atomic_store(&data->running, false);
    pw_main_loop_quit(data->loop);
    return NULL;
}

static void usage(FILE *out)
{
    fprintf(out,
            "Okay Hermes native PipeWire wake listener\n\n"
            "Usage: okay-hermes-wake-listener [--hermes-home PATH] [options]\n\n"
            "Options:\n"
            "  --hermes-home PATH    Hermes root; defaults to HERMES_HOME or ~/.hermes.\n"
            "  --model PATH          Override ONNX wakeword model path.\n"
            "  --activation-config PATH  Override config path passed to post-wake Python handler.\n"
            "  --handler-command CMD Command to run on activation; JSON is sent on stdin.\n"
            "  --threshold FLOAT     Wake probability threshold, default %.6f.\n"
            "  --consecutive N       Positive windows required, default %u.\n"
            "  --inference-interval-ms N  Minimum milliseconds between OrtRun calls.\n"
            "  --duration-seconds N  Stop after N seconds; 0 means run until interrupted.\n"
            "  --target OBJECT       PipeWire target object/node name or id.\n"
            "  --self-test           Load model, run one zero-input OrtRun, print score, exit.\n"
            "  --verbose             Print worker wake scores to stderr.\n"
            "  -h, --help            Show this help.\n",
            DEFAULT_THRESHOLD,
            DEFAULT_CONSECUTIVE);
}

static int parse_unsigned(const char *value, unsigned int *out)
{
    char *end = NULL;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX)
        return -1;
    *out = (unsigned int)parsed;
    return 0;
}

static int parse_float(const char *value, float *out)
{
    char *end = NULL;
    float parsed;

    errno = 0;
    parsed = strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !isfinite(parsed) || parsed < 0.0f)
        return -1;
    *out = parsed;
    return 0;
}

static int parse_options(int argc, char *argv[], struct listener_options *options)
{
    options->threshold = DEFAULT_THRESHOLD;
    options->consecutive = DEFAULT_CONSECUTIVE;
    options->duration_seconds = 0;
    options->inference_interval_ms = 250;
    options->target_object = NULL;
    options->model_path = NULL;
    options->handler_command = NULL;
    options->activation_config_path = NULL;
    options->hermes_home = NULL;
    options->verbose = false;
    options->self_test = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 1;
        }
        if (strcmp(argv[i], "--verbose") == 0) {
            options->verbose = true;
            continue;
        }
        if (strcmp(argv[i], "--self-test") == 0) {
            options->self_test = true;
            continue;
        }
        if (strcmp(argv[i], "--duration-seconds") == 0 && i + 1 < argc) {
            if (parse_unsigned(argv[++i], &options->duration_seconds) == 0)
                continue;
            fprintf(stderr, "invalid --duration-seconds value\n");
            return -1;
        }
        if (strcmp(argv[i], "--consecutive") == 0 && i + 1 < argc) {
            if (parse_unsigned(argv[++i], &options->consecutive) == 0 && options->consecutive > 0)
                continue;
            fprintf(stderr, "invalid --consecutive value\n");
            return -1;
        }
        if (strcmp(argv[i], "--inference-interval-ms") == 0 && i + 1 < argc) {
            if (parse_unsigned(argv[++i], &options->inference_interval_ms) == 0 && options->inference_interval_ms > 0)
                continue;
            fprintf(stderr, "invalid --inference-interval-ms value\n");
            return -1;
        }
        if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &options->threshold) == 0)
                continue;
            fprintf(stderr, "invalid --threshold value\n");
            return -1;
        }
        if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            options->target_object = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--hermes-home") == 0 && i + 1 < argc) {
            options->hermes_home = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            options->model_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--activation-config") == 0 && i + 1 < argc) {
            options->activation_config_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--handler-command") == 0 && i + 1 < argc) {
            options->handler_command = argv[++i];
            continue;
        }
        fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
        usage(stderr);
        return -1;
    }

    if (resolve_hermes_paths(options) < 0) {
        fprintf(stderr, "could not resolve Hermes paths\n");
        return -1;
    }
    return 0;
}

static int start_worker(struct listener_data *data)
{
    int rc = pthread_create(&data->worker, NULL, run_wakeword_worker, data);
    if (rc != 0) {
        fprintf(stderr, "failed to start worker thread: %s\n", strerror(rc));
        return -rc;
    }
    atomic_store(&data->worker_started, true);
    return 0;
}

static int start_duration_timer(struct listener_data *data, pthread_t *timer_thread)
{
    int rc;

    if (data->options.duration_seconds == 0)
        return 0;

    rc = pthread_create(timer_thread, NULL, duration_thread, data);
    if (rc != 0) {
        fprintf(stderr, "failed to start duration timer: %s\n", strerror(rc));
        return -rc;
    }
    return 0;
}

static struct pw_properties *stream_properties(const struct listener_options *options)
{
    struct pw_properties *props = pw_properties_new(PW_KEY_MEDIA_TYPE,
                                                    "Audio",
                                                    PW_KEY_MEDIA_CATEGORY,
                                                    "Capture",
                                                    PW_KEY_MEDIA_ROLE,
                                                    "Communication",
                                                    PW_KEY_NODE_NAME,
                                                    "okay-hermes-wake-listener",
                                                    PW_KEY_NODE_DESCRIPTION,
                                                    "Okay Hermes native wakeword listener",
                                                    NULL);
    if (props != NULL && options->target_object != NULL)
        pw_properties_set(props, PW_KEY_TARGET_OBJECT, options->target_object);
    return props;
}

static int run_self_test(struct wake_model *model)
{
    float *input = calloc(MODEL_SAMPLES, sizeof(float));
    float score = 0.0f;
    int rc;

    if (input == NULL)
        return 1;
    rc = run_model(model, input, &score);
    free(input);
    if (rc < 0)
        return 1;
    printf("{\"event\":\"self_test\",\"score\":%.9f}\n", score);
    return 0;
}

int main(int argc, char *argv[])
{
    struct listener_data data = {0};
    struct spa_pod *params[1];
    uint8_t buffer[1024];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
    struct pw_properties *props;
    pthread_t timer_thread = 0;
    bool timer_started = false;
    int parsed;
    int rc = 0;

    parsed = parse_options(argc, argv, &data.options);
    if (parsed != 0)
        return parsed > 0 ? 0 : 2;

    if (wake_model_init(&data.model, data.options.model_path) < 0)
        return 1;

    if (data.options.self_test) {
        rc = run_self_test(&data.model);
        wake_model_destroy(&data.model);
        return rc;
    }

    pw_init(&argc, &argv);
    atomic_store(&data.running, true);
    atomic_store(&data.worker_started, false);
    atomic_store(&data.negotiated_rate, MODEL_RATE);
    atomic_store(&data.negotiated_channels, 1);

    data.loop = pw_main_loop_new(NULL);
    if (data.loop == NULL) {
        fprintf(stderr, "failed to create PipeWire main loop\n");
        rc = 1;
        goto out_deinit;
    }

    if (ring_init(&data.ring, RING_SECONDS * MODEL_RATE) < 0) {
        fprintf(stderr, "failed to allocate audio ring buffer\n");
        rc = 1;
        goto out_loop;
    }

    pw_loop_add_signal(pw_main_loop_get_loop(data.loop), SIGINT, quit_signal, &data);
    pw_loop_add_signal(pw_main_loop_get_loop(data.loop), SIGTERM, quit_signal, &data);

    props = stream_properties(&data.options);
    if (props == NULL) {
        fprintf(stderr, "failed to allocate stream properties\n");
        rc = 1;
        goto out_ring;
    }

    data.stream = pw_stream_new_simple(pw_main_loop_get_loop(data.loop),
                                       "okay-hermes-wake-listener",
                                       props,
                                       &stream_events,
                                       &data);
    if (data.stream == NULL) {
        fprintf(stderr, "failed to create PipeWire stream\n");
        rc = 1;
        goto out_ring;
    }

    params[0] = spa_format_audio_raw_build(&builder,
                                           SPA_PARAM_EnumFormat,
                                           &SPA_AUDIO_INFO_RAW_INIT(.format = SPA_AUDIO_FORMAT_F32));

    if (pw_stream_connect(data.stream,
                          PW_DIRECTION_INPUT,
                          PW_ID_ANY,
                          PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS,
                          (const struct spa_pod **)params,
                          1) < 0) {
        fprintf(stderr, "failed to connect PipeWire capture stream\n");
        rc = 1;
        goto out_stream;
    }

    if (start_worker(&data) < 0) {
        rc = 1;
        goto out_stream;
    }

    if (start_duration_timer(&data, &timer_thread) < 0) {
        rc = 1;
        goto out_stream;
    }
    timer_started = data.options.duration_seconds != 0;

    pw_main_loop_run(data.loop);

out_stream:
    atomic_store(&data.running, false);
    if (data.stream != NULL)
        pw_stream_destroy(data.stream);
    if (atomic_load(&data.worker_started))
        pthread_join(data.worker, NULL);
    if (timer_started)
        pthread_join(timer_thread, NULL);
out_ring:
    ring_destroy(&data.ring);
out_loop:
    pw_main_loop_destroy(data.loop);
out_deinit:
    pw_deinit();
    wake_model_destroy(&data.model);
    return rc;
}
