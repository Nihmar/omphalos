#include "server/webui.hh"

#include <algorithm>
#include <cctype>

namespace omph::server::webui {
namespace {

constexpr std::string_view kImmutable = "public, max-age=31536000, immutable";
constexpr std::string_view kNoCache = "no-cache";

bool contains_ci(const std::string_view haystack, const std::string_view needle) {
    const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                                [](const char a, const char b) {
                                    return std::tolower((unsigned char) a) == (unsigned char) b;
                                });
    return it != haystack.end();
}

std::string_view trim(const std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

}  // namespace

bool etag_matches(const std::string_view if_none_match, const std::string_view etag) {
    for (size_t pos = 0; pos < if_none_match.size();) {
        const size_t comma = if_none_match.find(',', pos);
        std::string_view token = trim(if_none_match.substr(pos, comma == std::string_view::npos ? std::string_view::npos
                                                                                               : comma - pos));
        if (token == "*") {
            return true;
        }
        if (token.rfind("W/", 0) == 0) {
            token.remove_prefix(2);
        }
        if (token.size() == etag.size() + 2 && token.front() == '"' && token.back() == '"' &&
            token.substr(1, etag.size()) == etag) {
            return true;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        pos = comma + 1;
    }
    return false;
}

std::string cache_control_for(const std::string_view name) {
    // The names whose contents change on every build: they must be revalidated,
    // everything else carries a content hash in its name (llama.cpp's rule).
    if (name == "index.html" || name == "sw.js" || name == "manifest.webmanifest" || name == "build.json" ||
        name == "_app/version.json" || name == "version.json") {
        return std::string(kNoCache);
    }
    return std::string(kImmutable);
}

Resolution resolve(const std::string_view path, const std::string_view if_none_match,
                   const std::string_view accept_encoding) {
    Resolution res;
    std::string_view name = path;
    if (!name.empty() && name.front() == '/') {
        name.remove_prefix(1);
    }
    if (name.empty()) {
        name = "index.html";
    }
    if (const size_t query = name.find('?'); query != std::string_view::npos) {
        name = name.substr(0, query);
    }
    const Asset * asset = find(name);
    if (asset == nullptr) {
        return res;  // NotFound
    }
    res.asset = asset;
    res.cache_control = cache_control_for(asset->name);
    res.isolate = asset->name == "index.html";
    if (asset->gzip && !contains_ci(accept_encoding, "gzip")) {
        res.status = Resolution::Status::GzipRequired;
        return res;
    }
    res.status = etag_matches(if_none_match, asset->etag) ? Resolution::Status::NotModified
                                                          : Resolution::Status::Serve;
    return res;
}

}  // namespace omph::server::webui
