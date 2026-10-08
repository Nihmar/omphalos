# Embeds a llama.cpp web UI asset tree into the server (#378).
#
#   cmake -DWEBUI_DIR=<dir> -DOUT_CC=<file> -DOUT_HH=<file>
#         -DTEMPLATE_DIR=<engine/cmake> -P embed_webui.cmake
#
# Every file under WEBUI_DIR becomes one exact route at its relative path
# ("index.html", "_app/immutable/..."), embedded byte for byte with its
# SHA-256 as the ETag. A file whose bytes start with the gzip magic (1f 8b)
# keeps `Content-Encoding: gzip` when served: the source may be either a plain
# build tree (llama.cpp's tools/ui/dist) or the gzip staging directory its
# build makes (build.../tools/ui/ui-gzip/_gzip). A missing or empty WEBUI_DIR
# emits an empty table, so a build without the assets still compiles and
# answers 404 at "/".
#
# The tests run this same script on a fixture tree (engine/tests/fixtures/webui).

cmake_minimum_required(VERSION 3.21)

set(WEBUI_DIR    "" CACHE STRING "the asset tree to embed (empty: no web UI)")
set(OUT_CC       "" CACHE STRING "the generated .cc")
set(OUT_HH       "" CACHE STRING "the generated .hh")
set(TEMPLATE_DIR "" CACHE STRING "where webui_assets.{cc,hh}.in live")

if(NOT OUT_CC OR NOT OUT_HH OR NOT TEMPLATE_DIR)
    message(FATAL_ERROR "embed_webui.cmake needs OUT_CC, OUT_HH and TEMPLATE_DIR")
endif()

function(mime_from_ext name out_var)
    string(FIND "${name}" "." ext REVERSE)
    if(ext GREATER -1)
        string(SUBSTRING "${name}" ${ext} -1 ext_full)
        string(SUBSTRING "${ext_full}" 1 -1 ext_str)
    else()
        set(ext_str "")
    endif()
    if(ext_str STREQUAL "html")
        set(m "text/html; charset=utf-8")
    elseif(ext_str STREQUAL "css")
        set(m "text/css")
    elseif(ext_str STREQUAL "js")
        set(m "application/javascript")
    elseif(ext_str STREQUAL "json")
        set(m "application/json")
    elseif(ext_str STREQUAL "webmanifest")
        set(m "application/manifest+json")
    elseif(ext_str STREQUAL "svg")
        set(m "image/svg+xml")
    elseif(ext_str STREQUAL "png")
        set(m "image/png")
    elseif(ext_str STREQUAL "jpg" OR ext_str STREQUAL "jpeg")
        set(m "image/jpeg")
    elseif(ext_str STREQUAL "ico")
        set(m "image/x-icon")
    elseif(ext_str STREQUAL "woff")
        set(m "font/woff")
    elseif(ext_str STREQUAL "woff2")
        set(m "font/woff2")
    elseif(ext_str STREQUAL "txt")
        set(m "text/plain; charset=utf-8")
    elseif(ext_str STREQUAL "map")
        set(m "application/json")
    else()
        set(m "application/octet-stream")
    endif()
    set(${out_var} "${m}" PARENT_SCOPE)
endfunction()

set(assets "")
if(EXISTS "${WEBUI_DIR}/index.html")
    file(GLOB_RECURSE assets LIST_DIRECTORIES false RELATIVE "${WEBUI_DIR}" "${WEBUI_DIR}/*")
    list(SORT assets)
endif()
list(LENGTH assets N_ASSETS)

set(ASSET_ARRAYS "")
set(ASSET_TABLE "")
set(idx 0)
foreach(f IN LISTS assets)
    file(READ "${WEBUI_DIR}/${f}" hex HEX)
    string(LENGTH "${hex}" hexlen)
    math(EXPR nbytes "${hexlen} / 2")
    if(nbytes EQUAL 0)
        set(bytes "0x00,")  # a placeholder byte: the table's size is 0
    else()
        string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
    endif()
    file(SHA256 "${WEBUI_DIR}/${f}" etag)
    mime_from_ext("${f}" mime)
    set(gzip "false")
    if(nbytes GREATER 0)
        string(SUBSTRING "${hex}" 0 4 magic)
        if(magic STREQUAL "1f8b")
            set(gzip "true")
        endif()
    endif()
    string(APPEND ASSET_ARRAYS "static const unsigned char asset_${idx}[] = {${bytes}};\n")
    string(APPEND ASSET_TABLE "    { \"${f}\", asset_${idx}, ${nbytes}, \"${etag}\", \"${mime}\", ${gzip} },\n")
    math(EXPR idx "${idx} + 1")
endforeach()

# A zero-length array is not valid C++: keep a placeholder when there is no
# asset, and let the span's size (0) hide it.
set(ARRAY_SIZE "${N_ASSETS}")
if(N_ASSETS EQUAL 0)
    set(ARRAY_SIZE 1)
    set(ASSET_TABLE "    { \"\", nullptr, 0, \"\", \"application/octet-stream\", false },\n")
endif()

configure_file("${TEMPLATE_DIR}/webui_assets.hh.in" "${OUT_HH}" @ONLY)
configure_file("${TEMPLATE_DIR}/webui_assets.cc.in" "${OUT_CC}" @ONLY)
message(STATUS "web UI: embedded ${N_ASSETS} asset(s) from ${WEBUI_DIR}")
