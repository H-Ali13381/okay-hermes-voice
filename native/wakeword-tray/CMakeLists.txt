cmake_minimum_required(VERSION 3.20)
project(okay_hermes_wakeword_tray LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTORCC OFF)
set(CMAKE_AUTOUIC OFF)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(okay-hermes-wakeword-tray main.cpp)
target_link_libraries(okay-hermes-wakeword-tray PRIVATE Qt6::Widgets)

target_compile_options(okay-hermes-wakeword-tray PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -O2>
)
