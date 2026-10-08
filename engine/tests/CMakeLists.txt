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
            ${OMPH_SRC}/server/openai.cc ${OMPH_SRC}/server/tools.cc ${OMPH_SRC}/model/sampler.cc ${OMPH_SRC}/model/hparams.cc)
target_include_directories(omph_format PUBLIC ${OMPH_SRC})
target_compile_options(omph_format PRIVATE -Wall -Wextra)

enable_testing()

# The web UI table and its request decisions (#378): the server's embed script
# run on a fixture tree, so the asset lookup, the cache policy and the gzip /
# ETag rules are covered without a socket.
set(OMPH_WEBUI_GEN ${CMAKE_CURRENT_BINARY_DIR}/generated/server)
# The fixture tree as a glob: an asset change re-runs the embed step. The
# manifest file makes an added or removed asset re-run the custom command too
# (#391).
file(GLOB_RECURSE OMPH_WEBUI_FIXTURE CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/webui/*")
set(OMPH_WEBUI_FIXTURE_MANIFEST ${OMPH_WEBUI_GEN}/webui_fixture_manifest.txt)
file(CONFIGURE OUTPUT "${OMPH_WEBUI_FIXTURE_MANIFEST}" CONTENT
     "web UI fixture:\n${OMPH_WEBUI_FIXTURE}\n" @ONLY)
add_custom_command(
    OUTPUT ${OMPH_WEBUI_GEN}/webui_assets.cc ${OMPH_WEBUI_GEN}/webui_assets.hh
    COMMAND ${CMAKE_COMMAND} -DWEBUI_DIR=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/webui
            -DOUT_CC=${OMPH_WEBUI_GEN}/webui_assets.cc -DOUT_HH=${OMPH_WEBUI_GEN}/webui_assets.hh
            -DTEMPLATE_DIR=${OMPH_SRC}/../cmake -P ${OMPH_SRC}/../cmake/embed_webui.cmake
    DEPENDS ${OMPH_SRC}/../cmake/embed_webui.cmake ${OMPH_SRC}/../cmake/webui_assets.cc.in
            ${OMPH_SRC}/../cmake/webui_assets.hh.in ${OMPH_WEBUI_FIXTURE_MANIFEST} ${OMPH_WEBUI_FIXTURE}
    COMMENT "Embedding the test web UI fixture" VERBATIM)
add_library(omph_webui STATIC ${OMPH_SRC}/server/webui.cc ${OMPH_WEBUI_GEN}/webui_assets.cc)
target_include_directories(omph_webui PUBLIC ${OMPH_SRC} ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_options(omph_webui PRIVATE -Wall -Wextra)

foreach(t test_repack test_gguf test_omph test_tokenizer test_chat test_openai test_sampler test_hparams
           test_ngram test_argmax test_server_access test_tools)
    add_executable(${t} ${t}.cc)
    target_link_libraries(${t} PRIVATE omph_format)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

add_executable(test_webui test_webui.cc)
target_link_libraries(test_webui PRIVATE omph_webui)
target_compile_options(test_webui PRIVATE -Wall -Wextra)
add_test(NAME test_webui COMMAND test_webui)
