// omph-gguf-info — dump GGUF metadata and the tensor table (PLAN.md §9).
//
// usage: omph-gguf-info <model.gguf> [--tensor <name>] [--all]
#include "format/gguf.hh"

#include <cstdio>
#include <cstring>
#include <string_view>
#include <unordered_map>

namespace {

void print_value(const omph::gguf::Value & v) {
    switch (v.type) {
        case omph::gguf::ValueType::STRING:
            std::printf("\"%s\"", v.s.c_str());
            break;
        case omph::gguf::ValueType::FLOAT32:
        case omph::gguf::ValueType::FLOAT64:
            std::printf("%g", v.f);
            break;
        case omph::gguf::ValueType::BOOL:
            std::printf("%s", v.u ? "true" : "false");
            break;
        case omph::gguf::ValueType::INT8:
        case omph::gguf::ValueType::INT16:
        case omph::gguf::ValueType::INT32:
        case omph::gguf::ValueType::INT64:
            std::printf("%lld", (long long) v.i);
            break;
        case omph::gguf::ValueType::ARRAY:
            std::printf("[%llu values%s]", (unsigned long long) v.array_len,
                        v.truncated ? ", truncated" : "");
            break;
        default:
            std::printf("%llu", (unsigned long long) v.u);
            break;
    }
}

void print_metadata(const omph::gguf::File & f, std::string_view key) {
    if (const omph::gguf::Value * v = f.find(key)) {
        std::printf("  %-45s = ", std::string(key).c_str());
        print_value(*v);
        std::printf("\n");
    }
}

void print_tensor(const omph::gguf::TensorInfo & t) {
    const omph::gguf::TypeInfo * info = omph::gguf::type_info(t.type);
    std::printf("  %-40s %-10s ne=[", t.name.c_str(), info ? info->name : "?");
    for (size_t i = 0; i < t.ne.size(); ++i) {
        std::printf("%s%llu", i ? ", " : "", (unsigned long long) t.ne[i]);
    }
    std::printf("] bytes=%llu offset=%llu\n", (unsigned long long) t.nbytes,
                (unsigned long long) t.offset);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--tensor <name>] [--all]\n", argv[0]);
        return 2;
    }
    std::string_view tensor_name;
    bool all = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--tensor" && i + 1 < argc) {
            tensor_name = argv[++i];
        } else if (arg == "--all") {
            all = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    try {
        omph::gguf::File f(argv[1]);
        std::printf("file: %s\n", argv[1]);
        std::printf("gguf version: %u, metadata: %zu, tensors: %zu, data offset: %llu\n",
                    f.gguf_version(), f.metadata().size(), f.tensors().size(),
                    (unsigned long long) f.data_offset());

        std::printf("metadata (selected):\n");
        for (const char * key : {"general.architecture", "general.name", "qwen35.block_count",
                                 "qwen35.context_length", "qwen35.embedding_length",
                                 "qwen35.feed_forward_length", "qwen35.attention.head_count",
                                 "qwen35.attention.head_count_kv", "qwen35.attention.key_length",
                                 "qwen35.rope.dimension_count", "qwen35.rope.dimension_sections",
                                 "qwen35.rope.freq_base", "qwen35.ssm.conv_kernel",
                                 "qwen35.ssm.inner_size", "qwen35.ssm.state_size",
                                 "qwen35.ssm.time_step_rank", "qwen35.ssm.group_count",
                                 "qwen35.full_attention_interval", "qwen35.nextn_predict_layers",
                                 "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"}) {
            print_metadata(f, key);
        }

        std::printf("tensor table:\n");
        uint64_t total = 0;
        std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> by_type;  // type -> (count, bytes)
        for (const omph::gguf::TensorInfo & t : f.tensors()) {
            total += t.nbytes;
            auto & e = by_type[t.type];
            e.first += 1;
            e.second += t.nbytes;
            if (all || (!tensor_name.empty() && t.name == tensor_name)) {
                print_tensor(t);
            }
        }
        std::printf("  total: %zu tensors, %.2f GiB\n", f.tensors().size(),
                    (double) total / 1073741824.0);
        std::printf("  by type:\n");
        for (const auto & [type, counts] : by_type) {
            const omph::gguf::TypeInfo * info = omph::gguf::type_info(type);
            std::printf("    %-10s %5llu tensors  %8.1f MiB\n", info ? info->name : "?",
                        (unsigned long long) counts.first, (double) counts.second / 1048576.0);
        }
        if (!tensor_name.empty()) {
            if (const omph::gguf::TensorInfo * t = f.tensor(tensor_name)) {
                std::printf("tensor %s:\n", std::string(tensor_name).c_str());
                print_tensor(*t);
            } else {
                std::printf("tensor %s: not found\n", std::string(tensor_name).c_str());
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
