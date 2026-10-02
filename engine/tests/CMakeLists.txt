# CPU-only tests of the format layer (#105): no HIP, no ROCm, so they run on
# any Linux machine and in CI.
#
#   cmake -S engine/tests -B engine/build-tests && cmake --build engine/build-tests -j
#   ctest --test-dir engine/build-tests --output-on-failure
cmake_minimum_required(VERSION 3.21)
project(omphalos_cpu_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(OMPH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_library(omph_format STATIC ${OMPH_SRC}/format/gguf.cc ${OMPH_SRC}/format/repack.cc
            ${OMPH_SRC}/format/omph.cc ${OMPH_SRC}/format/sha256.cc
            ${OMPH_SRC}/text/tokenizer.cc ${OMPH_SRC}/text/json.cc ${OMPH_SRC}/text/chat.cc
            ${OMPH_SRC}/server/openai.cc)
target_include_directories(omph_format PUBLIC ${OMPH_SRC})
target_compile_options(omph_format PRIVATE -Wall -Wextra)

enable_testing()
foreach(t test_repack test_gguf test_omph test_tokenizer test_chat test_openai)
    add_executable(${t} ${t}.cc)
    target_link_libraries(${t} PRIVATE omph_format)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
