cmake_minimum_required(VERSION 3.21)
project(omphalos LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx1200)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Per-thread default stream (#71): the legacy null stream synchronizes with every
# other stream, which turns the side stream that overlaps independent GEMVs into
# a full serialization.
add_compile_definitions(HIP_API_PER_THREAD_DEFAULT_STREAM=1)
add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-fgpu-default-stream=per-thread>)

find_package(hip REQUIRED)

# Warnings for every target, the library and the tools alike (#100).
add_library(omph_warnings INTERFACE)
target_compile_options(omph_warnings INTERFACE -Wall -Wextra)

add_library(omphalos SHARED
    src/format/gguf.cc
    src/format/omph.cc
    src/format/repack.cc
    src/format/sha256.cc
    src/text/tokenizer.cc
    src/text/json.cc
    src/text/chat.cc
    src/runtime/context.cc
    src/runtime/engine_api.cc
    src/runtime/matmul.cc
    src/runtime/options.cc
    src/model/hparams.cc
    src/model/runner.cc
    src/model/forward.cc
    src/model/layers.cc
    src/model/dispatch.cc
    src/model/mtp.cc
    src/model/dflash.cc
    src/model/generator.cc
    src/kernels/probe.hip
    src/kernels/dequant.hip
    src/kernels/elementwise.hip
    src/kernels/attn.hip
    src/kernels/gdn.hip
    src/kernels/gemv.hip
    src/kernels/gemm.hip
    src/kernels/dflash.hip
)
target_include_directories(omphalos
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(omphalos PRIVATE -O3)
target_link_libraries(omphalos PRIVATE hip::host omph_warnings)

# Vision on the CPU (#160): llama.cpp's mtmd from a CPU-only llama.cpp build
# (no GPU backend may load into this process). Optional:
#   -DOMPH_LLAMA_DIR=<llama.cpp source> -DOMPH_LLAMA_LIB=<that build's bin/ with libmtmd, libllama>
set(OMPH_LLAMA_DIR "" CACHE PATH "llama.cpp source tree, for vision (empty: no vision)")
set(OMPH_LLAMA_LIB "" CACHE PATH "the library directory of a CPU-only llama.cpp build")
if(OMPH_LLAMA_DIR)
    find_library(OMPH_MTMD_LIB mtmd PATHS ${OMPH_LLAMA_LIB} NO_DEFAULT_PATH REQUIRED)
    find_library(OMPH_LLAMA_SO llama PATHS ${OMPH_LLAMA_LIB} NO_DEFAULT_PATH REQUIRED)
    find_library(OMPH_GGML_SO ggml PATHS ${OMPH_LLAMA_LIB} NO_DEFAULT_PATH REQUIRED)
    find_library(OMPH_GGML_BASE_SO ggml-base PATHS ${OMPH_LLAMA_LIB} NO_DEFAULT_PATH REQUIRED)
    if(EXISTS ${OMPH_LLAMA_LIB}/libggml-hip.so OR EXISTS ${OMPH_LLAMA_LIB}/libggml-vulkan.so)
        message(FATAL_ERROR "OMPH_LLAMA_LIB has a GPU backend: vision needs a CPU-only llama.cpp build")
    endif()
    target_sources(omphalos PRIVATE src/vision/encoder.cc)
    target_include_directories(omphalos PRIVATE ${OMPH_LLAMA_DIR}/include ${OMPH_LLAMA_DIR}/ggml/include
                                                ${OMPH_LLAMA_DIR}/tools/mtmd)
    target_link_libraries(omphalos PRIVATE ${OMPH_MTMD_LIB} ${OMPH_LLAMA_SO} ${OMPH_GGML_SO} ${OMPH_GGML_BASE_SO})
    set_target_properties(omphalos PROPERTIES BUILD_RPATH ${OMPH_LLAMA_LIB} INSTALL_RPATH ${OMPH_LLAMA_LIB})
    target_compile_definitions(omphalos PUBLIC OMPH_VISION=1)
    message(STATUS "omphalos: vision with ${OMPH_LLAMA_LIB}")
endif()

add_executable(omph-capi-demo examples/capi_demo.c)
target_link_libraries(omph-capi-demo PRIVATE omphalos)
target_compile_options(omph-capi-demo PRIVATE -Wall -Wextra -std=c11)

add_executable(omph-generate src/tools/generate.cc)
target_include_directories(omph-generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-generate PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-server src/tools/server.cc src/server/http.cc src/server/openai.cc)
target_include_directories(omph-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-server PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-convert src/tools/convert.cc)
target_include_directories(omph-convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-convert PRIVATE omphalos omph_warnings)

add_executable(omph-tokenize src/tools/tokenize.cc)
target_include_directories(omph-tokenize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-tokenize PRIVATE omphalos omph_warnings)

add_executable(omph-gguf-info src/tools/gguf_info.cc)
target_include_directories(omph-gguf-info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-gguf-info PRIVATE omphalos omph_warnings)

add_executable(omph-device-info src/tools/device_info.cc)
target_link_libraries(omph-device-info PRIVATE omphalos omph_warnings)

add_executable(omph-dequant src/tools/dequant.cc)
target_include_directories(omph-dequant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-dequant PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-linear src/tools/linear.cc)
target_include_directories(omph-linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-linear PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-attn src/tools/attn.cc)
target_include_directories(omph-attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-attn PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-gdn src/tools/gdn.cc)
target_include_directories(omph-gdn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-gdn PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-graph-probe src/tools/graph_probe.cc)
target_include_directories(omph-graph-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-graph-probe PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-attn-bench src/tools/attn_bench.cc)
target_include_directories(omph-attn-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-attn-bench PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-run src/tools/run.cc)
target_include_directories(omph-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-run PRIVATE omphalos hip::host omph_warnings)

add_executable(omph-gemv-bench src/tools/gemv-bench.cc)
target_include_directories(omph-gemv-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(omph-gemv-bench PRIVATE omphalos hip::host omph_warnings)
