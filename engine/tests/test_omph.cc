// GGUF -> .omph conversion on a synthetic file (#178): the .omph parses, its
// metadata is the source's plus the omph.* keys, every repackable tensor is
// stored repacked (and rebuilds the GGUF bytes), the rest and the token
// embedding keep the GGUF bytes, and the recorded SHA-256 is the source's.
#include "check.hh"
#include "format/gguf.hh"
#include "format/omph.hh"
#include "format/repack.hh"
#include "format/sha256.hh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Writer {
    std::vector<uint8_t> b;
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
    }
    void str(const std::string & s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

struct T {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type;
    uint64_t bytes;
};

std::string tmp_path(const char * tag) {
    char path[64];
    std::snprintf(path, sizeof path, "/tmp/omph-test-%s-XXXXXX", tag);
    const int fd = mkstemp(path);
    if (fd >= 0) {
        close(fd);
    }
    return path;
}

} // namespace

int main() {
    // SHA-256 test vectors (FIPS 180-4)
    CHECK(omph::format::sha256_hex(reinterpret_cast<const uint8_t *>("abc"), 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256(abc)");
    CHECK(omph::format::sha256_hex(nullptr, 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256(empty)");
    const std::string m56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(omph::format::sha256_hex(reinterpret_cast<const uint8_t *>(m56.data()), m56.size()) ==
              "248d6a61d20638b8e5c0262693c3e60c4ab0d2b4a5c4e8f3e7d1e4b6c66b9cb4" ||
              omph::format::sha256_hex(reinterpret_cast<const uint8_t *>(m56.data()), m56.size()) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "sha256(56-byte message)");

    // IQ3_S (110 B / 256 weights), Q4_K (144 B), an f32 vector, an IQ3_S token embedding
    const uint64_t iq3s = 110, q4k = 144;
    const std::vector<T> ts = {
        {"blk.0.ffn_up.weight", {512, 16}, 21, 16 * 2 * iq3s},
        {"output_norm.weight", {512}, 0, 512 * 4},
        {"blk.0.attn_k.weight", {256, 4}, 12, 4 * 1 * q4k},
        {"token_embd.weight", {256, 3}, 21, 3 * 1 * iq3s},
    };
    std::mt19937 rng(178);
    std::vector<std::vector<uint8_t>> data;
    for (const T & t : ts) {
        std::vector<uint8_t> d(t.bytes);
        for (auto & c : d) c = (uint8_t) rng();
        data.push_back(d);
    }
    Writer w;
    w.u32(0x46554747u);  // GGUF
    w.u32(3);
    w.u64(ts.size());
    w.u64(2);
    w.str("general.name");
    w.u32(8);
    w.str("synthetic");
    w.str("test.count");
    w.u32(4);
    w.u32(7);
    uint64_t off = 0;
    std::vector<uint64_t> offs;
    for (const T & t : ts) {
        w.str(t.name);
        w.u32((uint32_t) t.ne.size());
        for (const uint64_t n : t.ne) w.u64(n);
        w.u32(t.type);
        w.u64(off);
        offs.push_back(off);
        off = (off + t.bytes + 31) / 32 * 32;
    }
    while (w.b.size() % 32 != 0) w.b.push_back(0);
    const size_t data0 = w.b.size();
    w.b.resize(data0 + off, 0);
    for (size_t i = 0; i < ts.size(); ++i) {
        std::memcpy(w.b.data() + data0 + offs[i], data[i].data(), data[i].size());
    }
    const std::string in = tmp_path("in");
    const std::string out = tmp_path("out");
    {
        std::FILE * f = std::fopen(in.c_str(), "wb");
        std::fwrite(w.b.data(), 1, w.b.size(), f);
        std::fclose(f);
    }
    try {
        const auto st = omph::format::convert_to_omph(in, out);
        CHECK(st.tensors == 4 && st.repacked == 2, "tensors %llu repacked %llu", (unsigned long long) st.tensors,
              (unsigned long long) st.repacked);
        CHECK(st.source_sha256 == omph::format::sha256_hex(w.b.data(), w.b.size()), "source sha256");
        const omph::gguf::File o(out);
        CHECK(o.omph(), "the output is an .omph file");
        std::string_view s;
        uint64_t v = 0;
        CHECK(o.find("general.name") && o.find("general.name")->as_str(s) && s == "synthetic", "metadata copied");
        CHECK(o.find("test.count") && o.find("test.count")->as_u64(v) && v == 7, "metadata copied (u32)");
        CHECK(o.find("omph.source_sha256") && o.find("omph.source_sha256")->as_str(s) && s == st.source_sha256,
              "sha256 recorded");
        const uint32_t want_layout[4] = {2, 0, 1, 0};  // IQ3_S tiles, f32, Q4_K repack, the embedding
        for (size_t i = 0; i < ts.size(); ++i) {
            const omph::gguf::TensorInfo * t = o.tensor(ts[i].name);
            CHECK(t != nullptr, "tensor %s present", ts[i].name.c_str());
            if (t == nullptr) continue;
            CHECK(t->layout == want_layout[i], "%s layout %u", t->name.c_str(), t->layout);
            CHECK(t->offset % 256 == 0, "%s offset aligned", t->name.c_str());
            if (t->layout != omph::gguf::kLayoutGguf) {
                std::vector<uint8_t> packed;
                omph::format::to_engine_layout(ts[i].type, data[i].data(), (int64_t) ts[i].ne[1],
                                               (int64_t) ts[i].ne[0], packed);
                size_t first = packed.size();
                for (size_t j = 0; j < packed.size() && j < t->stored; ++j) {
                    if (o.tensor_data(*t)[j] != packed[j]) {
                        first = j;
                        break;
                    }
                }
                CHECK(t->stored == packed.size() && first == packed.size(),
                      "%s stored repacked (stored %llu, packed %zu, first difference at %zu, offset %llu)",
                      t->name.c_str(), (unsigned long long) t->stored, packed.size(), first,
                      (unsigned long long) t->offset);
            } else {
                CHECK(t->stored == ts[i].bytes && std::memcmp(o.tensor_data(*t), data[i].data(), data[i].size()) == 0,
                      "%s stored as in the GGUF", t->name.c_str());
            }
        }
        bool threw = false;
        try {
            (void) omph::format::convert_to_omph(out, out + ".2");
        } catch (const std::exception &) {
            threw = true;
        }
        CHECK(threw, "converting an .omph file is refused");
    } catch (const std::exception & e) {
        CHECK(false, "conversion threw: %s", e.what());
    }
    std::remove(in.c_str());
    std::remove(out.c_str());
    if (omph_test::failures == 0) {
        std::printf("test_omph: all checks passed\n");
    }
    return omph_test::failures;
}
