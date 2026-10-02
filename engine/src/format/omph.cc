#include "format/omph.hh"

#include "format/gguf.hh"
#include "format/repack.hh"
#include "format/sha256.hh"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace omph::format {

namespace {

constexpr uint64_t kTensorAlign = 256;

void fail(const std::string & msg) { throw std::runtime_error("omph-convert: " + msg); }

class Writer {
public:
    explicit Writer(const std::string & path) : f_(std::fopen(path.c_str(), "wb")) {
        if (f_ == nullptr) {
            fail("cannot create " + path);
        }
    }
    ~Writer() {
        if (f_ != nullptr) {
            std::fclose(f_);
        }
    }
    void bytes(const void * p, const size_t n) {
        if (n > 0 && std::fwrite(p, 1, n, f_) != n) {
            fail("write failed");
        }
        pos_ += n;
    }
    template <class T>
    void pod(const T v) { bytes(&v, sizeof v); }
    void str(const std::string & s) {
        pod<uint64_t>(s.size());
        bytes(s.data(), s.size());
    }
    void pad_to(const uint64_t off) {
        static const uint8_t zero[4096] = {};
        while (pos_ < off) {
            const size_t n = (size_t) std::min<uint64_t>(off - pos_, sizeof zero);
            bytes(zero, n);
        }
    }
    uint64_t pos() const { return pos_; }
    void close() {
        if (std::fclose(f_) != 0) {
            f_ = nullptr;
            fail("close failed");
        }
        f_ = nullptr;
    }

private:
    std::FILE * f_;
    uint64_t pos_ = 0;
};

// GGUF value type ids (gguf::ValueType)
constexpr uint32_t kU32 = 4, kStr = 8, kArr = 9, kU64 = 10;

} // namespace

ConvertStats convert_to_omph(const std::string & in_path, const std::string & out_path,
                             const std::function<void(uint64_t, uint64_t)> & progress) {
    const gguf::File src(in_path);
    if (src.omph()) {
        fail(in_path + " is already an .omph file");
    }
    ConvertStats st;
    st.bytes_in = src.size();
    st.source_sha256 = sha256_hex(src.base(), src.size());

    // every tensor's layout and stored size, in table order
    const std::vector<gguf::TensorInfo> & ts = src.tensors();
    std::vector<uint32_t> layouts(ts.size(), gguf::kLayoutGguf);
    std::vector<uint64_t> stored(ts.size());
    std::vector<uint64_t> offsets(ts.size());
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const gguf::TensorInfo & t = ts[i];
        const int64_t bb = quant_block_bytes(t.type);
        const int64_t packed = bb > 0 && t.nbytes % (uint64_t) bb == 0
                                   ? repacked_bytes(t.type, (int64_t) (t.nbytes / (uint64_t) bb))
                                   : 0;
        // the token embedding stays as it is: the engine gathers its rows on the host
        if (packed > 0 && t.name != "token_embd.weight") {
            layouts[i] = gguf::kLayoutRepack;
            stored[i] = (uint64_t) packed;
        } else {
            stored[i] = t.nbytes;
        }
        offsets[i] = off;
        off = (off + stored[i] + kTensorAlign - 1) / kTensorAlign * kTensorAlign;
    }

    Writer w(out_path);
    w.pod<uint32_t>(0x48504D4Fu);  // "OMPH"
    w.pod<uint32_t>(3);            // the GGUF v3 container
    w.pod<uint64_t>(ts.size());
    w.pod<uint64_t>(src.kv_count() + 4);
    w.bytes(src.base() + src.kv_begin(), (size_t) (src.kv_end() - src.kv_begin()));
    w.str("omph.format_version");
    w.pod<uint32_t>(kU32);
    w.pod<uint32_t>(gguf::kOmphFormatVersion);
    w.str("omph.source_sha256");
    w.pod<uint32_t>(kStr);
    w.str(st.source_sha256);
    w.str("omph.tensor_layouts");
    w.pod<uint32_t>(kArr);
    w.pod<uint32_t>(kU32);
    w.pod<uint64_t>(layouts.size());
    w.bytes(layouts.data(), layouts.size() * sizeof(uint32_t));
    w.str("omph.tensor_bytes");
    w.pod<uint32_t>(kArr);
    w.pod<uint32_t>(kU64);
    w.pod<uint64_t>(stored.size());
    w.bytes(stored.data(), stored.size() * sizeof(uint64_t));
    for (size_t i = 0; i < ts.size(); ++i) {
        w.str(ts[i].name);
        w.pod<uint32_t>((uint32_t) ts[i].ne.size());
        for (const uint64_t n : ts[i].ne) {
            w.pod<uint64_t>(n);
        }
        w.pod<uint32_t>(ts[i].type);
        w.pod<uint64_t>(offsets[i]);
    }
    // the data section: aligned as the copied general.alignment (or the
    // default) says, which divides the 256-byte tensor alignment
    uint64_t alignment = 32;
    if (const gguf::Value * a = src.find("general.alignment")) {
        uint64_t v = 0;
        if (a->as_u64(v) && v > 0) {
            alignment = v;
        }
    }
    if (kTensorAlign % alignment != 0) {
        fail("general.alignment " + std::to_string(alignment) + " does not divide 256");
    }
    const uint64_t data_start = (w.pos() + alignment - 1) / alignment * alignment;
    w.pad_to(data_start);
    std::vector<uint8_t> packed;
    std::vector<uint8_t> rebuilt;
    for (size_t i = 0; i < ts.size(); ++i) {
        const gguf::TensorInfo & t = ts[i];
        w.pad_to(data_start + offsets[i]);
        const uint8_t * gguf_bytes = src.tensor_data(t);
        if (layouts[i] == gguf::kLayoutRepack) {
            const int64_t n_blocks = (int64_t) (t.nbytes / (uint64_t) quant_block_bytes(t.type));
            packed.clear();  // the layouts' padding is left unwritten: zeros, not the previous tensor
            if (!repack_any(t.type, gguf_bytes, n_blocks, packed) || packed.size() != stored[i]) {
                fail("repack failed for " + t.name);
            }
            // bit-exact: the stored layout must rebuild the GGUF bytes
            rebuilt.assign((size_t) t.nbytes, 0);
            if (!unrepack_any(t.type, packed.data(), n_blocks, rebuilt) || rebuilt.size() != t.nbytes ||
                std::memcmp(rebuilt.data(), gguf_bytes, (size_t) t.nbytes) != 0) {
                fail("the repacked " + t.name + " does not rebuild the GGUF bytes");
            }
            w.bytes(packed.data(), packed.size());
            ++st.repacked;
        } else {
            w.bytes(gguf_bytes, (size_t) t.nbytes);
        }
        ++st.tensors;
        if (progress) {
            progress(i + 1, ts.size());
        }
    }
    st.bytes_out = w.pos();
    w.close();
    return st;
}

} // namespace omph::format
