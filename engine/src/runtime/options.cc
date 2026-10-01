#include "runtime/options.hh"

#include <cstdlib>

namespace omph::runtime {

namespace {

bool flag(const char * name) { return std::getenv(name) != nullptr; }

} // namespace

EnvOptions EnvOptions::from_env() {
    EnvOptions o;
    o.kv_host = flag("OMPH_KV_HOST");
    o.kv_f32 = flag("OMPH_KV_F32");
    o.kv_k4 = flag("OMPH_KV_K4");
    if (const char * w = std::getenv("OMPH_KV_WINDOW")) {
        o.kv_window = std::atoll(w);
    }
    o.no_overlap = flag("OMPH_NO_OVERLAP");
    o.no_b4 = flag("OMPH_NO_B4");
    o.no_bf16_gemv = flag("OMPH_NO_BF16_GEMV");
    o.no_f16_cache = flag("OMPH_NO_F16_CACHE");
    o.host_argmax = flag("OMPH_HOST_ARGMAX");
    o.gdn_per_token = flag("OMPH_GDN_PER_TOKEN");
    o.skip_attn = flag("OMPH_SKIP_ATTN");
    o.skip_ffn = flag("OMPH_SKIP_FFN");
    o.skip_blocks = flag("OMPH_SKIP_BLOCKS");
    o.skip_gemv = flag("OMPH_SKIP_GEMV");
    if (const char * t = std::getenv("OMPH_SKIP_GEMV_TYPE")) {
        o.skip_gemv_type = std::atoi(t);
    }
    o.skip_stage = flag("OMPH_SKIP_STAGE");
    o.timing = flag("OMPH_TIMING");
    o.phases = flag("OMPH_PHASES");
    o.trace_alloc = flag("OMPH_TRACE_ALLOC");
    o.trace_f16 = flag("OMPH_TRACE_F16");
    o.trace_stage = flag("OMPH_TRACE_STAGE");
    o.spec_check = flag("OMPH_SPEC_CHECK");
    return o;
}

} // namespace omph::runtime
