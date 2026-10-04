// The model loader (#315): a GGUF whose shapes are not the ones the kernels
// hard-code (kernels/shapes.hh) is refused at load, naming the field, instead
// of failing on the first layer that runs -- and the loader still accepts the
// model it was written for.
#include "check.hh"
#include "format/gguf.hh"
#include "model/hparams.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// Little-endian GGUF writer (as in test_gguf.cc).
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

struct Meta {  // a u32 metadata entry
    std::string key;
    uint64_t value;
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> ne;
    uint64_t offset;
};

// A GGUF v3 file: header, the u32 metadata, the tensor table (all F32), data.
std::vector<uint8_t> make_file(const std::vector<Meta> & meta, const std::vector<Tensor> & tensors,
                               const uint64_t data_bytes) {
    Writer w;
    w.u32(0x46554747u);
    w.u32(3);
    w.u64(tensors.size());
    w.u64(meta.size());
    for (const Meta & m : meta) {
        w.str(m.key);
        w.u32(4);  // UINT32
        w.u32((uint32_t) m.value);
    }
    for (const Tensor & t : tensors) {
        w.str(t.name);
        w.u32((uint32_t) t.ne.size());
        for (const uint64_t d : t.ne) w.u64(d);
        w.u32(0);  // F32
        w.u64(t.offset);
    }
    while (w.b.size() % 32 != 0) w.u8(0);
    for (uint64_t i = 0; i < data_bytes; ++i) w.u8(0);
    return w.b;
}

std::string write_tmp(const std::vector<uint8_t> & bytes) {
    char path[] = "/tmp/omph-test-hparams-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || write(fd, bytes.data(), bytes.size()) != (ssize_t) bytes.size()) {
        std::perror("mkstemp/write");
        std::exit(2);
    }
    close(fd);
    return path;
}

// The default hyperparameters of the model (tiny dimensions: the loader's
// checks do not depend on the sizes, only on the shapes the kernels hard-code).
std::vector<Meta> meta_of(const int64_t n_head = 24, const int64_t head_dim = 256,
                          const int64_t n_rot = 64, const int64_t ssm_s = 128,
                          const int64_t conv_k = 4, const int64_t n_kh = 2,
                          const int64_t n_vh = 4, const int64_t inner = 8) {
    return {{"qwen35.embedding_length", 8},
            {"qwen35.block_count", 2},
            {"qwen35.attention.head_count", (uint64_t) n_head},
            {"qwen35.attention.head_count_kv", 4},
            {"qwen35.attention.key_length", (uint64_t) head_dim},
            {"qwen35.rope.dimension_count", (uint64_t) n_rot},
            {"qwen35.feed_forward_length", 16},
            {"qwen35.ssm.group_count", (uint64_t) n_kh},
            {"qwen35.ssm.time_step_rank", (uint64_t) n_vh},
            {"qwen35.ssm.state_size", (uint64_t) ssm_s},
            {"qwen35.ssm.inner_size", (uint64_t) inner},
            {"qwen35.ssm.conv_kernel", (uint64_t) conv_k}};
}

// token_embd.weight [n_embd, n_vocab] then output.weight, both F32.
std::vector<Tensor> tensors_of(const uint64_t n_vocab = 16, const uint64_t out_vocab = 16) {
    const uint64_t bytes = 8 * n_vocab * 4;
    return {{"token_embd.weight", {8, n_vocab}, 0}, {"output.weight", {8, out_vocab}, bytes}};
}

uint64_t data_bytes_of(const uint64_t n_vocab = 16, const uint64_t out_vocab = 16) {
    return 8 * n_vocab * 4 + 8 * out_vocab * 4;
}

// Loads the file: "" when the loader accepted it, else what it threw.
std::string load(const std::vector<Meta> & meta, const std::vector<Tensor> & tensors,
                 const uint64_t data_bytes = data_bytes_of()) {
    const std::string path = write_tmp(make_file(meta, tensors, data_bytes));
    std::string err;
    try {
        omph::gguf::File f(path);
        (void) omph::model::read_hparams(f);
    } catch (const std::exception & e) {
        err = e.what();
    }
    unlink(path.c_str());
    return err;
}

bool refused(const std::vector<Meta> & meta, const std::vector<Tensor> & tensors, const char * what) {
    const std::string err = load(meta, tensors);
    const bool ok = err.find(what) != std::string::npos;
    if (!ok) {
        std::printf("  (got \"%s\", expected \"%s\")\n", err.c_str(), what);
    }
    return ok;
}

} // namespace

int main() {
    {
        const std::string path = write_tmp(make_file(meta_of(), tensors_of(), data_bytes_of()));
        try {
            omph::gguf::File f(path);
            const omph::model::HParams h = omph::model::read_hparams(f);
            CHECK(h.n_embd == 8 && h.n_layer == 2 && h.n_vocab == 16 && h.n_head == 24 &&
                      h.n_head_kv == 4,
                  "values: %lld %lld %lld", (long long) h.n_embd, (long long) h.n_layer,
                  (long long) h.n_vocab);
            CHECK(h.head_dim == 256 && h.ssm_s == 128 && h.ssm_conv_k == 4 && h.eps == 1e-6,
                  "kernels' shapes: %lld %lld", (long long) h.head_dim, (long long) h.ssm_s);
        } catch (const std::exception & e) {
            CHECK(false, "the model's shapes were refused: %s", e.what());
        }
        unlink(path.c_str());
    }
    CHECK(refused(meta_of(32), tensors_of(), "query heads per KV head"),
          "32 query heads for 4 kv heads");
    CHECK(refused(meta_of(24, 128), tensors_of(), "head_dim"), "head_dim 128");
    CHECK(refused(meta_of(24, 256, 512), tensors_of(), "rope"), "rope longer than head_dim");
    CHECK(refused(meta_of(24, 256, 63), tensors_of(), "rope"), "odd rope");
    CHECK(refused(meta_of(24, 256, 64, 64), tensors_of(), "delta-net state"), "ssm state 64");
    CHECK(refused(meta_of(24, 256, 64, 128, 8), tensors_of(), "conv window"), "conv window 8");
    CHECK(refused(meta_of(24, 256, 64, 128, 4, 3), tensors_of(), "delta-net state"),
          "3 groups for 4 heads");
    CHECK(refused(meta_of(24, 256, 64, 128, 4, 2, 6), tensors_of(), "delta-net state"),
          "6 heads for 8 values");
    CHECK(refused(meta_of(), tensors_of(16, 8), "vocabulary"), "output head, other vocabulary");
    {
        std::vector<Meta> missing = meta_of();
        missing.erase(missing.begin() + 1);  // block_count
        CHECK(refused(missing, tensors_of(), "incomplete hyperparameters"), "missing block_count");
    }
    if (omph_test::failures == 0) {
        std::printf("test_hparams: ok\n");
    }
    return omph_test::failures;
}
