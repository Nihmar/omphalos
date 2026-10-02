// A minimal JSON value for the chat template and the server (#150): parse
// (RFC 8259), objects keep their keys' order, and dump() writes what
// Python's json.dumps(value, ensure_ascii=False) writes, i.e. the `tojson`
// of HF transformers' chat templates.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace omph::text {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    static Json boolean(bool b);
    static Json number(double v, bool integer);
    static Json string(std::string s);
    static Json array();
    static Json object();

    // Throws std::runtime_error with the byte offset on malformed input.
    static Json parse(std::string_view text);
    std::string dump() const;

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const { return b_; }
    double as_number() const { return num_; }
    bool number_is_integer() const { return integer_; }
    const std::string & as_string() const { return str_; }

    // arrays
    const std::vector<Json> & items() const { return arr_; }
    void push(Json v) { arr_.push_back(std::move(v)); }
    size_t size() const { return type_ == Type::Array ? arr_.size() : obj_.size(); }

    // objects (insertion order); get() returns null for a missing key
    const std::vector<std::pair<std::string, Json>> & members() const { return obj_; }
    const Json * find(std::string_view key) const;
    const Json & get(std::string_view key) const;
    bool has(std::string_view key) const { return find(key) != nullptr; }
    void set(std::string key, Json v);

private:
    Type type_ = Type::Null;
    bool b_ = false;
    double num_ = 0.0;
    bool integer_ = false;
    std::string str_;
    std::vector<Json> arr_;
    std::vector<std::pair<std::string, Json>> obj_;
};

} // namespace omph::text
