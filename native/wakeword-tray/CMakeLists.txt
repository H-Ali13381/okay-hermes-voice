cmake_minimum_required(VERSION 3.20)
project(okay_hermes_wakeword_tray LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC OFF)
set(CMAKE_AUTOUIC OFF)

find_package(Qt6 REQUIRED COMPONENTS Widgets DBus)
find_package(KF6PulseAudioQt REQUIRED)

add_executable(okay-hermes-wakeword-tray main.cpp tray_state.h)
target_link_libraries(okay-hermes-wakeword-tray PRIVATE Qt6::Widgets Qt6::DBus KF6::PulseAudioQt)

target_compile_options(okay-hermes-wakeword-tray PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -O2>
)
