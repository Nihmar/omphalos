// The web UI table and the request decision (#378), without a socket: the
// fixture tree under fixtures/webui is embedded by the same script the server
// build runs (engine/cmake/embed_webui.cmake).
#include "check.hh"
#include "server/webui.hh"

#include <string>

namespace {

using omph::server::webui::Asset;
using omph::server::webui::Resolution;
using omph::server::webui::assets;
using omph::server::webui::cache_control_for;
using omph::server::webui::etag_matches;
using omph::server::webui::resolve;

Resolution::Status st(const std::string & path, const std::string & inm = "",
                      const std::string & enc = "gzip, deflate") {
    return resolve(path, inm, enc).status;
}

std::string quoted(const std::string_view etag) { return "\"" + std::string(etag) + "\""; }

void test_table() {
    CHECK(assets().size() == 7, "the fixture tree's seven files, sorted (%zu)", assets().size());
    const Asset * index = omph::server::webui::find("index.html");
    CHECK(index != nullptr, "the index is in the table");
    CHECK(index != nullptr && index->type == "text/html; charset=utf-8", "the index's type");
    CHECK(index != nullptr && !index->gzip, "the plain fixture is not gzip");
    const Asset * empty = omph::server::webui::find("empty.txt");
    CHECK(empty != nullptr && empty->size == 0 && empty->type == "text/plain; charset=utf-8",
          "a zero-byte asset embeds as an empty entry (#391)");
    const Asset * ico = omph::server::webui::find("favicon.ico");
    CHECK(ico != nullptr && ico->type == "image/x-icon", "the icon's type");
    const Asset * gz = omph::server::webui::find("_app/gz.js");
    CHECK(gz != nullptr && gz->gzip, "the gzip magic is detected");
    CHECK(omph::server::webui::find("nope") == nullptr, "an unknown name is not in the table");
}

void test_index() {
    const Resolution r = resolve("/");
    CHECK(r.status == Resolution::Status::Serve, "the root serves the index");
    CHECK(r.asset != nullptr && r.asset->name == "index.html", "the asset behind /");
    CHECK(r.isolate, "the index is served as an isolated context (COEP/COOP)");
    CHECK(r.cache_control == "no-cache", "the index is revalidated");
    const Resolution named = resolve("/index.html");
    CHECK(named.status == Resolution::Status::Serve && named.asset == r.asset, "/index.html is the same");
    CHECK(resolve("/").isolate, "and isolated");
}

void test_hashed_assets() {
    const Resolution r = resolve("/_app/bundle.js");
    CHECK(r.status == Resolution::Status::Serve, "a hashed asset is served");
    CHECK(r.cache_control.starts_with("public,"), "hashed assets are immutable: %s", r.cache_control.c_str());
    CHECK(!r.isolate, "only the index is isolated");
    CHECK(resolve("/_app/bundle.js?v=1").status == Resolution::Status::Serve, "the query string is ignored");
    CHECK(resolve("/_app/").status == Resolution::Status::NotFound, "a directory is not an asset");
    CHECK(resolve("/../etc/passwd").status == Resolution::Status::NotFound, "no traversal");
    CHECK(resolve("/no-such-file.js").status == Resolution::Status::NotFound, "an unknown file is a 404, not the index");
}

void test_cache_policy() {
    CHECK(cache_control_for("index.html") == "no-cache", "the index");
    CHECK(cache_control_for("sw.js") == "no-cache", "the service worker");
    CHECK(cache_control_for("manifest.webmanifest") == "no-cache", "the manifest");
    CHECK(cache_control_for("_app/version.json") == "no-cache", "the version");
    CHECK(cache_control_for("_app/bundle.js").starts_with("public,"), "a hashed name");
}

void test_gzip() {
    CHECK(st("/_app/gz.js", "", "gzip") == Resolution::Status::Serve, "a gzip asset to a gzip client");
    CHECK(st("/_app/gz.js", "", "gzip, deflate, br") == Resolution::Status::Serve, "in a list");
    CHECK(st("/_app/gz.js", "", "GZIP") == Resolution::Status::Serve, "case-insensitive");
    CHECK(st("/_app/gz.js", "", "") == Resolution::Status::GzipRequired, "no Accept-Encoding: 415");
    CHECK(st("/_app/gz.js", "", "identity") == Resolution::Status::GzipRequired, "a client that cannot gunzip: 415");
    CHECK(st("/_app/bundle.js", "", "") == Resolution::Status::Serve, "a plain asset needs no gzip");
}

void test_etag() {
    const Asset * a = omph::server::webui::find("_app/bundle.js");
    CHECK(a != nullptr, "the fixture asset exists");
    const std::string etag(a->etag);
    CHECK(a->etag.size() == 64, "the ETag is the SHA-256 hex");
    CHECK(etag_matches(quoted(etag), etag), "the exact ETag");
    CHECK(etag_matches("W/" + quoted(etag), etag), "a weak validator (llama.cpp's rule too)");
    CHECK(etag_matches("\"other\", " + quoted(etag), etag), "inside a list");
    CHECK(etag_matches("*", etag), "a wildcard");
    CHECK(!etag_matches("\"other\"", etag), "another ETag");
    CHECK(!etag_matches("", etag), "no header");
    CHECK(!etag_matches(quoted(etag) + "x", etag), "a suffix is not the ETag");
    CHECK(st("/_app/bundle.js", quoted(etag)) == Resolution::Status::NotModified, "a matching request is a 304");
    CHECK(st("/_app/bundle.js", quoted(etag)) != Resolution::Status::Serve, "and not a 200");
    CHECK(st("/_app/bundle.js", "\"other\"") == Resolution::Status::Serve, "another ETag is a 200");
    // gzip is decided before 304: a client that cannot gunzip never matches
    CHECK(st("/_app/gz.js", quoted(omph::server::webui::find("_app/gz.js")->etag), "") ==
              Resolution::Status::GzipRequired,
          "the gzip check comes first");
}

}  // namespace

int main() {
    test_table();
    test_index();
    test_hashed_assets();
    test_cache_policy();
    test_gzip();
    test_etag();
    if (omph_test::failures == 0) {
        std::printf("test_webui: ok\n");
    }
    return omph_test::failures;
}
