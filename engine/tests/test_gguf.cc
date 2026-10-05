// The GGUF parser on synthetic files: a well-formed one parses (metadata,
// tensors, data offsets), and every malformed one is rejected with an
// exception instead of a fault or a silent desync (#52, #91).
#include "check.hh"
#include "format/gguf.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// Little-endian GGUF writer.
struct Writer {
    std::vector<uint8_t> b;
    void u8(uint8_t v) { b.push_back(v); }
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

struct Tensor {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type;
    uint64_t offset;
};

// A GGUF v3 file: header, one u32 key, one string key, the tensor table, data.
std::vector<uint8_t> make_file(const std::vector<Tensor> & tensors, const uint64_t data_bytes,
                               const uint32_t magic = 0x46554747u, const uint32_t version = 3,
                               const uint32_t value_type = 4) {
    Writer w;
    w.u32(magic);
    w.u32(version);
    w.u64(tensors.size());
    w.u64(2);
    w.str("test.count");
    w.u32(value_type);
    w.u32(7);
    w.str("general.name");
    w.u32(8);
    w.str("synthetic");
    for (const Tensor & t : tensors) {
        w.str(t.name);
        w.u32((uint32_t) t.ne.size());
        for (const uint64_t d : t.ne) w.u64(d);
        w.u32(t.type);
        w.u64(t.offset);
    }
    while (w.b.size() % 32 != 0) w.u8(0);  // default alignment
    for (uint64_t i = 0; i < data_bytes; ++i) w.u8((uint8_t) i);
    return w.b;
}

std::string write_tmp(const std::vector<uint8_t> & bytes) {
    char path[] = "/tmp/omph-test-gguf-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || write(fd, bytes.data(), bytes.size()) != (ssize_t) bytes.size()) {
        std::perror("mkstemp/write");
        std::exit(2);
    }
    close(fd);
    return path;
}

// True when parsing throws a runtime_error whose message contains `what`.
bool rejects(const std::vector<uint8_t> & bytes, const char * what) {
    const std::string path = write_tmp(bytes);
    bool thrown = false;
    try {
        omph::gguf::File f(path);
    } catch (const std::runtime_error & e) {
        thrown = std::strstr(e.what(), what) != nullptr;
        if (!thrown) std::printf("  (threw \"%s\", expected \"%s\")\n", e.what(), what);
    }
    unlink(path.c_str());
    return thrown;
}

} // namespace

int main() {
    // #346: a tensor size whose 64-bit product would wrap is 0, so the file
    // check refuses it instead of accepting a truncated size
    CHECK(omph::gguf::type_nbytes(12, {256, 2}) == 288, "a Q4_K size");
    CHECK(omph::gguf::type_nbytes(12, {256, (1ull << 60) + 1}) == 0, "an overflowing Q4_K size");
    CHECK(omph::gguf::type_nbytes(12, {256, 0}) == 0, "a zero dimension");
    // F32 [4] at 0 (16 B), Q4_K [256, 2] at 32 (2 x 144 B)
    const std::vector<Tensor> good = {{"a.weight", {4}, 0, 0}, {"b.weight", {256, 2}, 12, 32}};
    const uint64_t data = 32 + 288;
    {
        const std::string path = write_tmp(make_file(good, data));
        try {
            omph::gguf::File f(path);
            uint64_t v = 0;
            std::string_view s;
            CHECK(f.gguf_version() == 3, "version %u", f.gguf_version());
            CHECK(f.find("test.count") != nullptr && f.find("test.count")->as_u64(v) && v == 7,
                  "u32 metadata");
            CHECK(f.find("general.name") != nullptr && f.find("general.name")->as_str(s) &&
                      s == "synthetic",
                  "string metadata");
            CHECK(f.tensors().size() == 2, "%zu tensors", f.tensors().size());
            const omph::gguf::TensorInfo * b = f.tensor("b.weight");
            CHECK(b != nullptr && b->nbytes == 288, "Q4_K [256, 2] is 288 bytes");
            CHECK(f.tensor("a.weight") != nullptr && f.tensor("a.weight")->nbytes == 16,
                  "F32 [4] is 16 bytes");
            CHECK(b != nullptr && f.tensor_data(*b) == f.base() + f.data_offset() + 32 &&
                      f.tensor_data(*b)[0] == 32,
                  "tensor data at data_offset + offset");
            CHECK(f.tensor("missing") == nullptr, "unknown tensor");
        } catch (const std::exception & e) {
            CHECK(false, "well-formed file rejected: %s", e.what());
        }
        unlink(path.c_str());
    }
    CHECK(rejects(make_file(good, data, 0x12345678u), "bad magic"), "bad magic");
    CHECK(rejects(make_file(good, data, 0x46554747u, 2), "unsupported GGUF version"),
          "version 2");
    CHECK(rejects(make_file(good, data, 0x46554747u, 3, 99), "unknown value type"),
          "unknown metadata value type");
    {
        std::vector<uint8_t> cut = make_file(good, data);
        cut.resize(40);  // inside the metadata
        CHECK(rejects(cut, "truncated"), "file cut in the header");
    }
    CHECK(rejects(make_file(good, 32 + 100), "beyond the end"),
          "tensor data past the end of the file");
    CHECK(rejects(make_file({{"x", {4}, 0, 1u << 20}}, 16), "beyond the end"),
          "tensor offset past the end");
    CHECK(rejects(make_file({{"x", {4}, 99, 0}}, 16), "unknown type"), "unknown tensor type");
    CHECK(rejects(make_file({{"x", {100, 2}, 12, 0}}, 1024), "whole number of blocks"),
          "Q4_K row not a multiple of 256");
    CHECK(rejects(make_file({{"x", {}, 0, 0}}, 16), "dimension count"), "zero dimensions");
    std::printf("test_gguf: %d failure(s)\n", omph_test::failures);
    return omph_test::failures == 0 ? 0 : 1;
}
