#include "runtime/options.hh"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace omph::runtime {

namespace {

bool flag(const char * name) { return std::getenv(name) != nullptr; }

} // namespace

EnvOptions EnvOptions::from_env() {
    EnvOptions o;
    o.kv_host = flag("OMPH_KV_HOST");
    if (const char * m = std::getenv("OMPH_HOST_STAGE_MIB")) {
        o.host_stage_mib = std::atoll(m) < 0 ? -1 : std::atoll(m);
    }
    o.kv_f32 = flag("OMPH_KV_F32");
    o.kv_k4 = flag("OMPH_KV_K4");
    if (const char * l = std::getenv("OMPH_KV_K4_LAYERS")) {
        o.kv_k4_layers = 0;  // "none" (or an empty list): every layer Q8
        for (const char * p = l; *p != '\0';) {
            char * end = nullptr;
            const long i = std::strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            if (i >= 0 && i < 64) {
                o.kv_k4_layers |= (uint64_t) 1 << i;
            }
            p = *end == ',' ? end + 1 : end;
        }
    }
    if (const char * w = std::getenv("OMPH_KV_WINDOW")) {
        o.kv_window = std::atoll(w);
    }
    o.overlap = flag("OMPH_OVERLAP");
    if (const char * s = std::getenv("OMPH_STAGE_MIB")) {
        o.stage_mib = std::atoi(s);
    }
    if (const char * g = std::getenv("OMPH_GEMM_MIN")) {
        if (*g != '\0') {
            o.gemm_min = std::atoi(g);
        }
    }
    o.gdn_serial = flag("OMPH_GDN_SERIAL");
    o.gdn_exact = flag("OMPH_GDN_EXACT");
    o.attn_scalar = flag("OMPH_ATTN_SCALAR");
    o.attn_dec_scalar = flag("OMPH_ATTN_DEC_SCALAR");
    o.no_fused_gemm = flag("OMPH_NO_FUSED_GEMM");
    o.no_b4 = flag("OMPH_NO_B4");
    o.spec_pointmass = flag("OMPH_SPEC_POINTMASS");
    o.no_group = flag("OMPH_NO_GROUP");
    o.no_swiglu_gemm = flag("OMPH_NO_SWIGLU_GEMM");
    if (const char * v = std::getenv("OMPH_MTP_WINDOW")) {
        o.mtp_window = std::atoll(v);
    }
    if (const char * v = std::getenv("OMPH_DRAFT_VOCAB")) {
        o.draft_vocab = std::atoll(v);
    }
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
    o.check_finite = flag("OMPH_CHECK_FINITE");
    if (const char * b = std::getenv("OMPH_TEST_BAD_SIDE")) {
        o.test_bad_side = std::atoi(b);
    }
    if (const char * m = std::getenv("OMPH_TEST_MROPE")) {
        o.test_mrope = std::string(m) == "swap" ? 1 : std::string(m) == "flat" ? 2 : 0;
    }
    if (const char * p = std::getenv("OMPH_NGRAM")) {
        o.ngram = std::atoi(p);
    }
    if (const char * p = std::getenv("OMPH_NGRAM_MIN")) {
        o.ngram_min = std::atoi(p);
    }
    if (const char * p = std::getenv("OMPH_DFLASH_KEEP")) {
        o.dflash_keep = (float) std::atof(p);
    }
    if (const char * p = std::getenv("OMPH_DFLASH_PMIN")) {
        o.dflash_pmin = (float) std::atof(p);
    }
    if (const char * q = std::getenv("OMPH_TEST_Q8ACT")) {
        o.test_q8act = std::atoi(q);
    }
    // Ablations give wrong results with valid timings (the header says so): a
    // stale one in the environment must not look like an engine bug. Name the
    // ones that are on, once, where they are parsed (#315).
    std::string off;
    const struct {
        const char * name;
        bool on;
    } ablations[] = {
        {"OMPH_SKIP_ATTN", o.skip_attn},        {"OMPH_SKIP_FFN", o.skip_ffn},
        {"OMPH_SKIP_BLOCKS", o.skip_blocks},    {"OMPH_SKIP_GEMV", o.skip_gemv},
        {"OMPH_SKIP_GEMV_TYPE", o.skip_gemv_type >= 0},
        {"OMPH_SKIP_STAGE", o.skip_stage},      {"OMPH_TEST_BAD_SIDE", o.test_bad_side > 0},
        {"OMPH_TEST_MROPE", o.test_mrope != 0}, {"OMPH_TEST_Q8ACT", o.test_q8act > 0},
    };
    for (const auto & a : ablations) {
        if (a.on) {
            off += off.empty() ? "" : " ";
            off += a.name;
        }
    }
    if (!off.empty()) {
        std::fprintf(stderr, "omphalos: ablation active: %s -- results are not the engine's\n",
                     off.c_str());
    }
    return o;
}

} // namespace omph::runtime
