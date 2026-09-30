cmake_minimum_required(VERSION 3.21)
project(omphalos LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx1200)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

add_library(omphalos SHARED
    src/format/gguf.cc
    src/runtime/context.cc
    src/runtime/matmul.cc
    src/kernels/probe.hip
    src/kernels/dequant.hip
    src/kernels/elementwise.hip
    src/kernels/attn.hip
)
target_include_directories(omphalos
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(omphalos PRIVATE -O3 -Wall -Wextra)
target_link_libraries(omphalos PRIVATE roc::hipblaslt hip::host)

add_executable(omph-gguf-info src/tools/gguf_info.cc)
target_include_directories(omph-gguf-info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-gguf-info PRIVATE omphalos)

add_executable(omph-device-info src/tools/device_info.cc)
target_link_libraries(omph-device-info PRIVATE omphalos)

add_executable(omph-dequant src/tools/dequant.cc)
target_include_directories(omph-dequant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-dequant PRIVATE omphalos hip::host)

add_executable(omph-linear src/tools/linear.cc)
target_include_directories(omph-linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-linear PRIVATE omphalos hip::host roc::hipblaslt)

add_executable(omph-attn src/tools/attn.cc)
target_include_directories(omph-attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-attn PRIVATE omphalos hip::host roc::hipblaslt)
