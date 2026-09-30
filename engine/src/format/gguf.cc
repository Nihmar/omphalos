#include "format/gguf.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

namespace omph::gguf {
namespace {

constexpr uint32_t kMagic = 0x46554747u;  // "GGUF" little-endian

[[noreturn]] void fail(const std::string & msg) {
    throw std::runtime_error("gguf: " + msg);
}

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if ((size_t) (end - p) < n) {
            fail("truncated file");
        }
    }
    uint8_t u8() {
        need(1);
        return *p++;
    }
    uint16_t u16() {
        need(2);
        uint16_t v;
        std::memcpy(&v, p, 2);
        p += 2;
        return v;
    }
    uint32_t u32() {
        need(4);
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    uint64_t u64() {
        need(8);
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    std::string str() {
        const uint64_t n = u64();
        need((size_t) n);
        std::string s(reinterpret_cast<const char *>(p), (size_t) n);
        p += n;
        return s;
    }
};

Value read_value(Cursor & c, ValueType t) {
    Value v;
    v.type = t;
    switch (t) {
        case ValueType::UINT8: v.u = c.u8(); break;
        case ValueType::INT8: v.i = (int8_t) c.u8(); break;
        case ValueType::UINT16: v.u = c.u16(); break;
        case ValueType::INT16: v.i = (int16_t) c.u16(); break;
        case ValueType::UINT32: v.u = c.u32(); break;
        case ValueType::INT32: v.i = (int32_t) c.u32(); break;
        case ValueType::FLOAT32: {
            const uint32_t bits = c.u32();
            float f;
            std::memcpy(&f, &bits, 4);
            v.f = f;
            break;
        }
        case ValueType::BOOL: v.u = c.u8() != 0; break;
        case ValueType::STRING: v.s = c.str(); break;
        case ValueType::UINT64: v.u = c.u64(); break;
        case ValueType::INT64: v.i = (int64_t) c.u64(); break;
        case ValueType::FLOAT64: {
            const uint64_t bits = c.u64();
            double d;
            std::memcpy(&d, &bits, 8);
            v.f = d;
            break;
        }
        case ValueType::ARRAY: {
            const ValueType elem_type = (ValueType) c.u32();
            const uint64_t n = c.u64();
            if (elem_type == ValueType::ARRAY) {
                fail("nested arrays are not supported");
            }
            v.array_len = n;
            constexpr uint64_t kKeep = 256;  // keep big arrays cheap
            for (uint64_t k = 0; k < n; ++k) {
                Value elem = read_value(c, elem_type);
                if (k < kKeep) {
                    v.array.push_back(std::move(elem));
                }
            }
            v.truncated = n > kKeep;
            break;
        }
    }
    return v;
}

constexpr TypeInfo kTypes[] = {
    {0, 1, 4, "F32"},      {1, 1, 2, "F16"},        {2, 32, 18, "Q4_0"},    {3, 32, 20, "Q4_1"},
    {6, 32, 22, "Q5_0"},   {7, 32, 24, "Q5_1"},     {8, 32, 34, "Q8_0"},    {9, 32, 36, "Q8_1"},
    {10, 256, 84, "Q2_K"}, {11, 256, 110, "Q3_K"},  {12, 256, 144, "Q4_K"}, {13, 256, 176, "Q5_K"},
    {14, 256, 210, "Q6_K"}, {15, 256, 292, "Q8_K"},
    {16, 256, 66, "IQ2_XXS"}, {17, 256, 74, "IQ2_XS"}, {18, 256, 98, "IQ3_XXS"},
    {19, 256, 50, "IQ1_S"}, {20, 32, 18, "IQ4_NL"},   {21, 256, 110, "IQ3_S"},
    {22, 256, 82, "IQ2_S"}, {23, 256, 136, "IQ4_XS"},
    {24, 1, 1, "I8"},      {25, 1, 2, "I16"},        {26, 1, 4, "I32"},      {27, 1, 8, "I64"},
    {28, 1, 8, "F64"},     {29, 256, 56, "IQ1_M"},   {30, 1, 2, "BF16"},
    {34, 256, 54, "TQ1_0"}, {35, 256, 66, "TQ2_0"},
};

} // namespace

bool Value::as_u64(uint64_t & out) const {
    if (type == ValueType::UINT8 || type == ValueType::UINT16 || type == ValueType::UINT32 ||
        type == ValueType::UINT64 || type == ValueType::BOOL) {
        out = u;
        return true;
    }
    if (type == ValueType::INT8 || type == ValueType::INT16 || type == ValueType::INT32 ||
        type == ValueType::INT64) {
        out = (uint64_t) i;
        return true;
    }
    return false;
}

bool Value::as_str(std::string_view & out) const {
    if (type != ValueType::STRING) {
        return false;
    }
    out = s;
    return true;
}

const TypeInfo * type_info(uint32_t type) {
    for (const TypeInfo & t : kTypes) {
        if (t.type == type) {
            return &t;
        }
    }
    return nullptr;
}

uint64_t type_nbytes(uint32_t type, const std::vector<uint64_t> & ne) {
    const TypeInfo * info = type_info(type);
    if (info == nullptr || ne.empty() || ne[0] % info->block != 0) {
        return 0;
    }
    uint64_t n = ne[0] / info->block * info->bytes;
    for (size_t d = 1; d < ne.size(); ++d) {
        n *= ne[d];
    }
    return n;
}

File::File(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fail("cannot open " + path);
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        fail("fstat failed for " + path);
    }
    size_ = (size_t) st.st_size;
    void * map = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        fail("mmap failed for " + path);
    }
    base_ = static_cast<const uint8_t *>(map);

    Cursor c{base_, base_ + size_};
    if (c.u32() != kMagic) {
        fail("bad magic (not a GGUF file)");
    }
    version_ = c.u32();
    if (version_ != 3) {
        fail("unsupported GGUF version " + std::to_string(version_));
    }
    const uint64_t tensor_count = c.u64();
    const uint64_t kv_count = c.u64();

    kv_.reserve(std::min<uint64_t>(kv_count, 4096));
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.str();
        const ValueType t = (ValueType) c.u32();
        kv_index_.emplace(key, kv_.size());
        kv_.emplace_back(std::move(key), read_value(c, t));
    }

    tensors_.reserve(std::min<uint64_t>(tensor_count, 4096));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo ti;
        ti.name = c.str();
        const uint32_t n_dims = c.u32();
        if (n_dims == 0 || n_dims > 4) {
            fail("bad dimension count for tensor " + ti.name);
        }
        ti.ne.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            ti.ne[d] = c.u64();
        }
        ti.type = c.u32();
        ti.offset = c.u64();
        ti.nbytes = type_nbytes(ti.type, ti.ne);
        tensor_index_.emplace(ti.name, tensors_.size());
        tensors_.push_back(std::move(ti));
    }

    uint64_t alignment = 32;  // ggml default
    if (const Value * a = find("general.alignment")) {
        uint64_t v = 0;
        if (a->as_u64(v) && v > 0) {
            alignment = v;
        }
    }
    const uint64_t offset = (uint64_t) (c.p - base_);
    data_offset_ = (offset + alignment - 1) / alignment * alignment;
    if (data_offset_ > size_) {
        fail("data section starts beyond the end of the file");
    }
    // Every tensor must have a known type and lie inside the mapping: a
    // truncated file would otherwise fault at upload instead of failing here.
    const uint64_t data_bytes = size_ - data_offset_;
    for (const TensorInfo & t : tensors_) {
        if (type_info(t.type) == nullptr) {
            fail("unknown type " + std::to_string(t.type) + " for tensor " + t.name);
        }
        if (t.nbytes == 0) {
            fail("row size is not a whole number of blocks for tensor " + t.name);
        }
        if (t.offset > data_bytes || t.nbytes > data_bytes - t.offset) {
            fail("data of tensor " + t.name + " extends beyond the end of the file");
        }
    }
}

File::~File() {
    if (base_ != nullptr) {
        ::munmap(const_cast<uint8_t *>(base_), size_);
    }
}

const Value * File::find(std::string_view key) const {
    const auto it = kv_index_.find(std::string(key));
    return it == kv_index_.end() ? nullptr : &kv_[it->second].second;
}

const TensorInfo * File::tensor(std::string_view name) const {
    const auto it = tensor_index_.find(std::string(name));
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

} // namespace omph::gguf
