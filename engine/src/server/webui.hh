// The llama.cpp web UI, served from omph-server (#378).
//
// The UI is the static bundle llama-server embeds (tools/ui in its tree): a
// SvelteKit app whose index.html links its hashed assets relatively and whose
// API calls are the OpenAI endpoints plus /props and /slots. This file holds
// the *decision* for one request -- which asset, whether the client may have
// it, whether it is unchanged, how it may be cached -- with no sockets, so the
// CPU tests cover it; server.cc turns the decision into a response.
//
// The asset table itself is generated at build time by
// engine/cmake/embed_webui.cmake from the tree named by OMPH_WEBUI_DIR
// (llama.cpp's tools/ui/dist, or its ui-gzip/_gzip stage); an empty tree makes
// every request NotFound.
#pragma once

#include <string>
#include <string_view>

#include "server/webui_assets.hh"

namespace omph::server::webui {

// What to answer for one asset request.
struct Resolution {
    enum class Status {
        Serve,         // 200: `asset`, with `cache_control` and `isolate`
        NotModified,   // 304: the client's If-None-Match matches the ETag
        GzipRequired,  // 415: the asset is gzip and the client does not accept it
        NotFound,      // no such asset: the caller continues its own routing
    };
    Status status = Status::NotFound;
    const Asset * asset = nullptr;
    // The Cache-Control to send: the hashed assets never change under a given
    // name and are immutable, the names that do change (index.html, sw.js,
    // manifest, version.json) must be revalidated.
    std::string cache_control;
    // index.html gets COEP/COOP: the UI is served as an isolated context
    // (llama.cpp does the same; some of its features require it).
    bool isolate = false;
};

// `path` is the request path without the query string ("/", "/index.html",
// "/_app/immutable/bundle.x.js"). `if_none_match` and `accept_encoding` are
// those headers as received (empty when absent). Both "/" and "/index.html"
// serve the index; every other asset is an exact match on its name (a missing
// file is a 404, never the index: the UI's routes are client-side but its
// files are not).
Resolution resolve(std::string_view path, std::string_view if_none_match = {},
                   std::string_view accept_encoding = {});

// True when, and only when, `if_none_match` names `etag` (the quotes and a
// weak "W/" included; "*" counts too): the condition of a 304.
bool etag_matches(std::string_view if_none_match, std::string_view etag);

// The Cache-Control for an asset name: "no-cache" for the few names whose
// contents change on every build, the immutable one for everything else.
std::string cache_control_for(std::string_view name);

}  // namespace omph::server::webui
