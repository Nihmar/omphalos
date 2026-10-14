// The on-disk conversation cache (#437), on the CPU: the index round trip, the
// prefix matching with the Generator's rule, the blob header and its FNV, the
// model-key filter and the LRU eviction. No GPU: the class only moves bytes.
#include "check.hh"
#include "model/diskcache.hh"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

using omph::model::DiskCache;

std::string temp_dir(const char * name) {
    const std::string dir = std::string("/tmp/omph-test-") + name;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return dir;
}

// The prefix rule the Generator passes in: `e.tokens` (and the image digests)
// agree with `want` for the first p positions.
bool fits(const DiskCache::Entry & e, const int64_t p, const std::vector<int32_t> & want,
          const std::vector<std::string> & want_images) {
    if (p > (int64_t) e.tokens.size() || p > (int64_t) want.size()) {
        return false;
    }
    for (int64_t i = 0; i < p; ++i) {
        if (e.tokens[(size_t) i] != want[(size_t) i]) {
            return false;
        }
        if (e.tokens[(size_t) i] >= 0) {
            continue;
        }
        const size_t idx = (size_t) (-1 - e.tokens[(size_t) i]);
        if (idx >= e.images.size() || idx >= want_images.size() || e.images[idx] != want_images[idx]) {
            return false;
        }
    }
    return true;
}

DiskCache::Entry entry(const std::vector<int32_t> & tokens, const std::vector<int64_t> & points,
                       const std::vector<std::string> & images, const std::string & model) {
    DiskCache::Entry e;
    e.n = (int64_t) tokens.size();
    e.tokens = tokens;
    e.images = images;
    e.kv_bytes = 64;
    e.cp_bytes = 32;
    e.points = points;
    e.file = omph::model::disk_file_name(model, e.n, tokens);
    return e;
}

void test_round_trip() {
    const std::string dir = temp_dir("diskcache");
    const std::string model = "abcdef01";
    std::vector<uint8_t> blob(4096);
    for (size_t i = 0; i < blob.size(); ++i) {
        blob[i] = (uint8_t) (i * 7 + 3);
    }
    const std::vector<int32_t> tokens{5, 6, 7, 8, 9, 10};
    {
        auto done = std::atomic<bool>(false);
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "open");
        CHECK(cache.enabled(), "enabled");
        cache.save(entry(tokens, {4, 6}, {}, model), blob.data(), (int64_t) blob.size(), &done);
        for (int i = 0; i < 500 && !done.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(done.load(), "the writer finished");
        CHECK(cache.entries() == 1, "one entry");
    }
    {  // a new process: the index and the blob come back
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "reopen");
        const std::vector<int32_t> want{5, 6, 7, 8, 9, 10, 11, 12};
        const DiskCache::Entry * e = cache.match(
            [&](const DiskCache::Entry & c, const int64_t p) { return fits(c, p, want, {}); });
        CHECK(e != nullptr, "the match found the entry");
        std::vector<uint8_t> got;
        CHECK(e != nullptr && cache.load(*e, got), "load");
        CHECK(got == blob, "the blob is identical");
        // a prefix that diverges must not match
        const std::vector<int32_t> other{5, 6, 99, 8, 9, 10, 11, 12};
        CHECK(cache.match([&](const DiskCache::Entry & c, const int64_t p) { return fits(c, p, other, {}); }) ==
                  nullptr,
              "a divergent prefix does not match");
        // a position past the entry cannot match either
        CHECK(cache.match([&](const DiskCache::Entry &, const int64_t p) { return p > 100; }) == nullptr,
              "no point beyond the entry");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_corrupt_blob() {
    const std::string dir = temp_dir("diskcache-corrupt");
    const std::string model = "abcdef01";
    std::vector<uint8_t> blob(1024, 0x5a);
    const std::vector<int32_t> tokens{1, 2, 3, 4};
    {
        auto done = std::atomic<bool>(false);
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "open");
        cache.save(entry(tokens, {4}, {}, model), blob.data(), (int64_t) blob.size(), &done);
        for (int i = 0; i < 500 && !done.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(done.load(), "written");
    }
    const std::string file = dir + "/" + omph::model::disk_file_name(model, 4, tokens);
    {  // flip one byte of the blob: the load must refuse it (the FNV in the header)
        FILE * f = std::fopen(file.c_str(), "r+b");
        CHECK(f != nullptr, "open the blob");
        if (f != nullptr) {
            (void) std::fseek(f, (long) DiskCache::kHeaderBytes + 10, SEEK_SET);
            const int byte = 0x00;
            CHECK(std::fwrite(&byte, 1, 1, f) == 1, "flip a byte");
            (void) std::fclose(f);
        }
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "reopen");
        const std::vector<int32_t> want{1, 2, 3, 4, 5, 6};
        const DiskCache::Entry * e = cache.match(
            [&](const DiskCache::Entry & c, const int64_t p) { return fits(c, p, want, {}); });
        CHECK(e != nullptr, "the entry is still listed");
        std::vector<uint8_t> got;
        CHECK(e == nullptr || !cache.load(*e, got), "a corrupted blob is refused");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_model_key_and_eviction() {
    const std::string dir = temp_dir("diskcache-evict");
    std::vector<uint8_t> blob(1 << 20, 1);
    const std::vector<int32_t> a{1, 2, 3, 4, 5};
    const std::vector<int32_t> b{1, 2, 3, 9, 9};
    {
        auto d1 = std::atomic<bool>(false);
        auto d2 = std::atomic<bool>(false);
        DiskCache cache;
        CHECK(cache.open(dir, "aaaa0000", 1), "open");  // a 1 MiB budget
        cache.save(entry(a, {5}, {}, "aaaa0000"), blob.data(), (int64_t) blob.size(), &d1);
        for (int i = 0; i < 500 && !d1.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(d1.load(), "the first write finished");
        cache.save(entry(b, {5}, {}, "aaaa0000"), blob.data(), (int64_t) blob.size(), &d2);
        for (int i = 0; i < 500 && !d2.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(d2.load(), "the second write finished");
        CHECK(cache.entries() == 1, "the 1 MiB budget keeps one 1 MiB blob (got %zu)", cache.entries());
    }
    {  // a file of another model is dropped at open
        auto d3 = std::atomic<bool>(false);
        DiskCache cache;
        CHECK(cache.open(dir, "bbbb0000", 1), "open bbbb");
        cache.save(entry(a, {5}, {}, "bbbb0000"), blob.data(), (int64_t) blob.size(), &d3);
        for (int i = 0; i < 500 && !d3.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(d3.load(), "the bbbb write finished");
        CHECK(cache.entries() == 1, "bbbb has its entry");
    }
    {
        DiskCache cache;
        CHECK(cache.open(dir, "cccc0000", 1), "open cccc");
        CHECK(cache.entries() == 0, "another model's entries are dropped (got %zu)", cache.entries());
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_images() {
    const std::string dir = temp_dir("diskcache-images");
    const std::string model = "abcdef01";
    std::vector<uint8_t> blob(64, 2);
    const std::vector<int32_t> tokens{1, -1, 2, 3};  // -1: the first image's rows
    const std::vector<std::string> images{"deadbeef"};
    {
        auto done = std::atomic<bool>(false);
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "open");
        cache.save(entry(tokens, {4}, images, model), blob.data(), (int64_t) blob.size(), &done);
        for (int i = 0; i < 500 && !done.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    {
        DiskCache cache;
        CHECK(cache.open(dir, model, 64), "reopen");
        const std::vector<int32_t> want{1, -1, 2, 3, 4, 5, 6};
        const DiskCache::Entry * same = cache.match(
            [&](const DiskCache::Entry & c, const int64_t p) { return fits(c, p, want, images); });
        CHECK(same != nullptr, "the same image matches");
        const std::vector<std::string> different{"cafebabe"};
        CHECK(cache.match([&](const DiskCache::Entry & c, const int64_t p) { return fits(c, p, want, different); }) ==
                  nullptr,
              "a different image does not match");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

} // namespace

int main() {
    test_round_trip();
    test_corrupt_blob();
    test_model_key_and_eviction();
    test_images();
    std::printf("%s\n", omph_test::failures == 0 ? "test_diskcache: ok" : "test_diskcache: FAILURES");
    return omph_test::failures == 0 ? 0 : 1;
}
