// GGUF reader for the omphalos engine (PLAN.md §9.1), and for the engine's
// own .omph files (#178, PLAN.md §8.4): the same container with the magic
// "OMPH", the source GGUF's metadata copied verbatim, four omph.* keys (format
// version, source SHA-256, every tensor's layout and stored bytes), and the
// weights already in the layouts the kernels read (omph-convert writes them).
//
// Minimal on purpose: mmap the file, parse the header, the metadata KV block
// and the tensor table. Only what this one model needs — no generality.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omph::gguf {

// ggml metadata value types (GGUF v3).
enum class ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5, FLOAT32 = 6,
    BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

struct Value {
    ValueType type = ValueType::UINT64;
    uint64_t u = 0;      // unsigned integers and bool
    int64_t i = 0;       // signed integers
    double f = 0.0;      // floats
    std::string s;       // strings
    uint64_t array_len = 0;
    bool truncated = false;      // big arrays keep only the first elements
    std::vector<Value> array;
    ValueType elem_type = ValueType::UINT8;  // arrays: the element type
    uint64_t array_offset = 0;   // arrays: file offset of the first element (File::*_array)

    bool as_u64(uint64_t & out) const;
    bool as_str(std::string_view & out) const;
};

// Block layout of a ggml type (see ggml-common.h).
struct TypeInfo {
    uint32_t type;
    uint32_t block;
    uint32_t bytes;
    const char * name;
};

// Returns nullptr when the type is not in the table.
const TypeInfo * type_info(uint32_t type);

// Byte size of a tensor with these dims (ne[0] fastest); 0 when unknown.
uint64_t type_nbytes(uint32_t type, const std::vector<uint64_t> & ne);

// .omph tensor layouts (omph.tensor_layouts)
enum Layout : uint32_t {
    kLayoutGguf = 0,    // the GGUF bytes as they are
    kLayoutRepack = 1,  // format/repack.hh's layout of the type (PLAN.md §8.3)
};
constexpr uint32_t kOmphFormatVersion = 1;

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;   // ne[0] is the fastest / contiguous dimension
    uint32_t type = 0;          // ggml type id
    uint64_t offset = 0;        // relative to the start of the data section
    uint64_t nbytes = 0;        // computed from ne and type (the GGUF bytes)
    uint32_t layout = kLayoutGguf;  // .omph: how the stored bytes are laid out
    uint64_t stored = 0;        // bytes in the file: nbytes, or the layout's size
};

class File {
public:
    // Throws std::runtime_error on malformed files.
    explicit File(const std::string & path);
    ~File();
    File(const File &) = delete;
    File & operator=(const File &) = delete;

    uint32_t gguf_version() const { return version_; }
    // An .omph file (magic "OMPH") rather than a GGUF one.
    bool omph() const { return omph_; }
    // The byte range of the metadata KV block (omph-convert copies it).
    uint64_t kv_begin() const { return kv_begin_; }
    uint64_t kv_end() const { return kv_end_; }
    uint64_t kv_count() const { return kv_.size(); }
    const std::vector<std::pair<std::string, Value>> & metadata() const { return kv_; }
    const Value * find(std::string_view key) const;
    // Every element of an array (the metadata keeps only the first 256 of a
    // big one): strings, or integers of any width. Throw if `key` is missing
    // or of another type.
    std::vector<std::string> string_array(std::string_view key) const;
    std::vector<int64_t> int_array(std::string_view key) const;
    const std::vector<TensorInfo> & tensors() const { return tensors_; }
    const TensorInfo * tensor(std::string_view name) const;
    uint64_t data_offset() const { return data_offset_; }
    size_t size() const { return size_; }
    const uint8_t * base() const { return base_; }
    const uint8_t * tensor_data(const TensorInfo & t) const {
        return base_ + data_offset_ + t.offset;
    }

private:
    const uint8_t * base_ = nullptr;
    size_t size_ = 0;
    uint32_t version_ = 0;
    bool omph_ = false;
    uint64_t kv_begin_ = 0;
    uint64_t kv_end_ = 0;
    uint64_t data_offset_ = 0;
    std::vector<std::pair<std::string, Value>> kv_;
    std::unordered_map<std::string, size_t> kv_index_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

} // namespace omph::gguf
