#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstring>

struct ggml_cuda_mmq_ids_q8_reuse_cache {
    bool ready = false;
    int  device = -1;

    const ggml_tensor * src1 = nullptr;
    const ggml_tensor * ids  = nullptr;
    cudaStream_t stream = nullptr;

    ggml_type type_x = GGML_TYPE_COUNT;
    int64_t token0 = 0;
    int64_t ne10 = 0;
    int64_t ne10_padded = 0;
    int64_t ne11 = 0;
    int64_t ne12 = 0;
    int64_t ne12_chunk = 0;
    int64_t ne_get_rows = 0;
    int64_t ne02 = 0;
    int64_t n_expert_used = 0;
    int64_t ids_si1 = 0;
    int64_t ids_sis1 = 0;

    char * src1_q8_1 = nullptr;
    size_t src1_q8_1_actual = 0;

    int32_t * ids_src1 = nullptr;
    size_t ids_src1_actual = 0;

    int32_t * ids_dst = nullptr;
    size_t ids_dst_actual = 0;

    int32_t * expert_bounds = nullptr;
    size_t expert_bounds_actual = 0;
};

static ggml_cuda_mmq_ids_q8_reuse_cache g_mmq_ids_q8_reuse_cache[GGML_CUDA_MAX_DEVICES];

struct ggml_cuda_mmq_precomp_q8_reuse_cache {
    bool ready = false;
    int  device = -1;

    const ggml_tensor * src1_key = nullptr;
    const ggml_tensor * ids_key  = nullptr;
    cudaStream_t stream = nullptr;

    ggml_type type_x = GGML_TYPE_COUNT;
    int64_t expert_low = 0;
    int64_t expert_high = 0;
    int64_t ne10 = 0;
    int64_t ne10_padded = 0;
    int64_t compact_rows = 0;

    char * src1_q8_1 = nullptr;
    size_t src1_q8_1_actual = 0;
};

static ggml_cuda_mmq_precomp_q8_reuse_cache g_mmq_precomp_q8_reuse_cache[GGML_CUDA_MAX_DEVICES];

static void ggml_cuda_mmq_device_realloc(const int device, void ** ptr, size_t * actual_size, const size_t required_size) {
    if (*ptr != nullptr && *actual_size >= required_size) {
        return;
    }

    if (*ptr != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
        *actual_size = 0;
    }

    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMalloc(ptr, required_size));
    *actual_size = required_size;
}

static bool ggml_cuda_mmq_ids_reuse_cacheable(const int cc, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_IDS_REUSE_Q8")) {
        return false;
    }

    if (!GGML_CUDA_CC_IS_AMD(cc) && !ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_IDS_REUSE_Q8_ALL_DEVICES")) {
        return false;
    }

    // MiniMax separate gate/up expert matmuls share this normalized activation.
    // Do not cache the down projection: its source is different and should
    // invalidate a pending gate/up cache to avoid cross-layer or cross-ubatch reuse.
    if (strstr(src1->name, "ffn_norm") == nullptr) {
        return false;
    }

    return strstr(dst->name, "ffn_moe_gate") != nullptr || strstr(dst->name, "ffn_moe_up") != nullptr;
}

static bool ggml_cuda_mmq_ids_reuse_matches(
        const ggml_cuda_mmq_ids_q8_reuse_cache & cache,
        const int device, const cudaStream_t stream, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_type type_x, const int64_t token0, const int64_t ne10, const int64_t ne10_padded,
        const int64_t ne11, const int64_t ne12, const int64_t ne12_chunk, const int64_t ne_get_rows,
        const int64_t ne02, const int64_t n_expert_used, const int64_t ids_si1, const int64_t ids_sis1) {
    return cache.ready &&
        cache.device == device &&
        cache.stream == stream &&
        cache.src1 == src1 &&
        cache.ids == ids &&
        cache.type_x == type_x &&
        cache.token0 == token0 &&
        cache.ne10 == ne10 &&
        cache.ne10_padded == ne10_padded &&
        cache.ne11 == ne11 &&
        cache.ne12 == ne12 &&
        cache.ne12_chunk == ne12_chunk &&
        cache.ne_get_rows == ne_get_rows &&
        cache.ne02 == ne02 &&
        cache.n_expert_used == n_expert_used &&
        cache.ids_si1 == ids_si1 &&
        cache.ids_sis1 == ids_sis1;
}

static void ggml_cuda_mmq_ids_reuse_log(const char * action, const int device, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_IDS_REUSE_Q8_LOG")) {
        return;
    }

    fprintf(stderr, "ggml_cuda_mmq_ids_reuse_q8: dev=%d %s src1=%s dst=%s\n",
        device, action, src1 ? src1->name : "(null)", dst ? dst->name : "(null)");
    fflush(stderr);
}

static bool ggml_cuda_mmq_precomp_reuse_cacheable(const int cc) {
    if (!ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_PRECOMP_REUSE_Q8")) {
        return false;
    }

    if (!GGML_CUDA_CC_IS_AMD(cc) && !ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_PRECOMP_REUSE_Q8_ALL_DEVICES")) {
        return false;
    }

    return true;
}

static bool ggml_cuda_mmq_precomp_reuse_matches(
        const ggml_cuda_mmq_precomp_q8_reuse_cache & cache,
        const int device, const cudaStream_t stream, const ggml_tensor * src1_key, const ggml_tensor * ids_key,
        const ggml_type type_x, const int64_t expert_low, const int64_t expert_high,
        const int64_t ne10, const int64_t ne10_padded, const int64_t compact_rows) {
    return cache.ready &&
        cache.device == device &&
        cache.stream == stream &&
        cache.src1_key == src1_key &&
        cache.ids_key == ids_key &&
        cache.type_x == type_x &&
        cache.expert_low == expert_low &&
        cache.expert_high == expert_high &&
        cache.ne10 == ne10 &&
        cache.ne10_padded == ne10_padded &&
        cache.compact_rows == compact_rows;
}

static void ggml_cuda_mmq_precomp_reuse_log(const char * action, const int device, const ggml_tensor * src1_key, const ggml_tensor * ids_key) {
    if (!ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_PRECOMP_REUSE_Q8_LOG")) {
        return;
    }

    fprintf(stderr, "ggml_cuda_mmq_precomp_reuse_q8: dev=%d %s src1=%s ids=%s\n",
        device, action, src1_key ? src1_key->name : "(null)", ids_key ? ids_key->name : "(null)");
    fflush(stderr);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static bool ggml_cuda_mul_mat_alloc_debug_should_log(const int device, int & count) {
    if (!ggml_cuda_mmq_env_enabled("GGML_CUDA_MUL_MAT_ALLOC_DEBUG")) {
        return false;
    }

    const int debug_device = ggml_cuda_mmq_env_int("GGML_CUDA_MUL_MAT_ALLOC_DEBUG_DEVICE", -1);
    if (debug_device >= 0 && debug_device != device) {
        return false;
    }

    const int debug_limit = ggml_cuda_mmq_env_int("GGML_CUDA_MUL_MAT_ALLOC_DEBUG_LIMIT", 128);
    if (debug_limit >= 0 && count >= debug_limit) {
        return false;
    }

    ++count;
    return true;
}

static void ggml_cuda_mul_mat_alloc_debug_log(
        const char * func, const int device, const char * label, const char * name, const size_t request_bytes,
        const size_t actual_bytes, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "%s: dev=%d alloc_debug %s %s request=%.2f MiB actual=%.2f MiB cudaMemGetInfo_failed=%d "
            "src0=%s src1=%s ids=%s dst=%s\n",
            func, device, label, name, request_bytes / 1048576.0, actual_bytes / 1048576.0, (int) err,
            src0 ? src0->name : "(null)", src1 ? src1->name : "(null)", ids ? ids->name : "(null)",
            dst ? dst->name : "(null)");
        fflush(stderr);
        return;
    }

    fprintf(stderr,
        "%s: dev=%d alloc_debug %s %s request=%.2f MiB actual=%.2f MiB free=%.2f MiB total=%.2f MiB "
        "src0=%s:%s ne=[%lld,%lld,%lld,%lld] src1=%s:%s ne=[%lld,%lld,%lld,%lld] ids=%s ne=[%lld,%lld,%lld,%lld] "
        "dst=%s:%s ne=[%lld,%lld,%lld,%lld]\n",
        func, device, label, name, request_bytes / 1048576.0, actual_bytes / 1048576.0,
        free_bytes / 1048576.0, total_bytes / 1048576.0,
        src0 ? src0->name : "(null)", src0 ? ggml_type_name(src0->type) : "(null)",
        src0 ? (long long) src0->ne[0] : 0LL, src0 ? (long long) src0->ne[1] : 0LL,
        src0 ? (long long) src0->ne[2] : 0LL, src0 ? (long long) src0->ne[3] : 0LL,
        src1 ? src1->name : "(null)", src1 ? ggml_type_name(src1->type) : "(null)",
        src1 ? (long long) src1->ne[0] : 0LL, src1 ? (long long) src1->ne[1] : 0LL,
        src1 ? (long long) src1->ne[2] : 0LL, src1 ? (long long) src1->ne[3] : 0LL,
        ids ? ids->name : "(null)",
        ids ? (long long) ids->ne[0] : 0LL, ids ? (long long) ids->ne[1] : 0LL,
        ids ? (long long) ids->ne[2] : 0LL, ids ? (long long) ids->ne[3] : 0LL,
        dst ? dst->name : "(null)", dst ? ggml_type_name(dst->type) : "(null)",
        dst ? (long long) dst->ne[0] : 0LL, dst ? (long long) dst->ne[1] : 0LL,
        dst ? (long long) dst->ne[2] : 0LL, dst ? (long long) dst->ne[3] : 0LL);
    fflush(stderr);
}

static bool ggml_cuda_mmq_env_device_list_contains(const char * name, const int device) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return true;
    }

    const char * cur = value;
    while (*cur != '\0') {
        while (*cur == ',' || *cur == ' ') {
            ++cur;
        }
        if (*cur == '\0') {
            break;
        }

        char * end = nullptr;
        const long parsed = strtol(cur, &end, 10);
        if (end != cur && parsed == device) {
            return true;
        }
        cur = end != cur ? end : cur + 1;
    }

    return false;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool no_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_NO_STREAM_K");
    const bool use_stream_k = !no_stream_k &&
                            ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc));

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const int device = ggml_cuda_get_device();
    static int alloc_debug_count = 0;

    if (!ids) {
        const int64_t s11_src1 = src1->nb[1] / ts_src1;
        const int64_t s12_src1 = src1->nb[2] / ts_src1;
        const int64_t s13_src1 = src1->nb[3] / ts_src1;
        const int64_t chunk_cols_env = ggml_cuda_mmq_env_int("GGML_CUDA_MMQ_NO_IDS_CHUNK_COLS", 0);
        const bool chunk_device_allowed = ggml_cuda_mmq_env_device_list_contains("GGML_CUDA_MMQ_NO_IDS_CHUNK_DEVICES", device);
        const int64_t chunk_cols = chunk_device_allowed && chunk_cols_env > 0 && chunk_cols_env < ne11 ? chunk_cols_env : ne11;
        const bool chunk_no_stream_k = chunk_cols < ne11 && ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_CHUNK_NO_STREAM_K");
        const int chunk_force_x = chunk_cols < ne11 ? ggml_cuda_mmq_env_int("GGML_CUDA_MMQ_CHUNK_FORCE_X", 0) : 0;

        for (int64_t col0 = 0; col0 < ne11; col0 += chunk_cols) {
            const int64_t ne11_chunk = col0 + chunk_cols > ne11 ? ne11 - col0 : chunk_cols;
            const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11_chunk*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
            ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
            if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "before", "src1_q8_1_no_ids",
                    nbytes_src1_q8_1, 0, src0, src1, ids, dst);
            }
            src1_q8_1.alloc(nbytes_src1_q8_1);
            if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "after", "src1_q8_1_no_ids",
                    nbytes_src1_q8_1, src1_q8_1.actual_size, src0, src1, ids, dst);
            }

            const float * src1_chunk_d = src1_d + col0*s11_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_chunk_d, nullptr, src1_q8_1.get(), src0->type, ne10,
                                       s11_src1, s12_src1, s13_src1, ne10_padded,
                                       ne11_chunk, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_chunk_d, nullptr, src1_q8_1.get(), src0->type, ne10,
                                       s11_src1, s12_src1, s13_src1, ne10_padded,
                                       ne11_chunk, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());

            // Stride depends on quantization format.
            const int64_t s12 = use_native_fp4 ?
                                    ne11_chunk * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :  // block_fp4_mmq holds 256 values
                                    ne11_chunk * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
            const int64_t s13 = ne12*s12;

            const mmq_args args = {
                src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d + col0*s1,
                ne00, ne01, ne11_chunk, s01, ne11_chunk, s1,
                ne02, ne12, s02, s12, s2,
                ne03, ne13, s03, s13, s3,
                use_stream_k && !chunk_no_stream_k, ne11_chunk, chunk_force_x};
            ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    GGML_ASSERT(ne1 == n_expert_used);

    const int64_t ids_chunk_tokens_env = ggml_cuda_mmq_env_int("GGML_CUDA_MMQ_IDS_CHUNK_TOKENS", 0);
    const bool ids_chunk_device_allowed = ggml_cuda_mmq_env_device_list_contains("GGML_CUDA_MMQ_IDS_CHUNK_DEVICES", device);
    const int64_t ids_chunk_tokens = ids_chunk_device_allowed && ids_chunk_tokens_env > 0 && ids_chunk_tokens_env < ne12 ?
        ids_chunk_tokens_env : ne12;
    const bool ids_chunk_no_stream_k = ids_chunk_tokens < ne12 && ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_CHUNK_NO_STREAM_K");
    const int ids_chunk_force_x = ids_chunk_tokens < ne12 ? ggml_cuda_mmq_env_int("GGML_CUDA_MMQ_CHUNK_FORCE_X", 0) : 0;

    for (int64_t token0 = 0; token0 < ne12; token0 += ids_chunk_tokens) {
        const int64_t ne12_chunk = token0 + ids_chunk_tokens > ne12 ? ne12 - token0 : ids_chunk_tokens;
        const int64_t ne_get_rows = ne12_chunk * n_expert_used;

        const size_t nbytes_src1_q8_1 = ne12_chunk*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);

        const int64_t ne11_flat = ne12_chunk*n_expert_used;
        const int64_t ne12_flat = 1;
        const int64_t ne13_flat = 1;

        ggml_cuda_pool_alloc<int32_t> ids_src1_local;
        ggml_cuda_pool_alloc<int32_t> ids_dst_local;
        ggml_cuda_pool_alloc<int32_t> expert_bounds_local;
        ggml_cuda_pool_alloc<char>    src1_q8_1_local;

        int32_t * ids_src1_ptr       = nullptr;
        int32_t * ids_dst_ptr        = nullptr;
        int32_t * expert_bounds_ptr  = nullptr;
        char    * src1_q8_1_ptr      = nullptr;

        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int ids_si1  = ids->nb[1] / ggml_element_size(ids);
        const int ids_sis1 = nb12 / nb11;

        ggml_cuda_mmq_ids_q8_reuse_cache & reuse_cache = g_mmq_ids_q8_reuse_cache[device];
        bool reuse_cacheable = ggml_cuda_mmq_ids_reuse_cacheable(cc, src1, dst);
        if (reuse_cacheable && reuse_cache.stream != nullptr && reuse_cache.stream != stream) {
            reuse_cacheable = false;
        }
        const bool reuse_hit = reuse_cacheable && ggml_cuda_mmq_ids_reuse_matches(
            reuse_cache, device, stream, src1, ids, src0->type, token0, ne10, ne10_padded, ne11, ne12, ne12_chunk,
            ne_get_rows, ne02, n_expert_used, ids_si1, ids_sis1);

        if (reuse_hit) {
            ids_src1_ptr      = reuse_cache.ids_src1;
            ids_dst_ptr       = reuse_cache.ids_dst;
            expert_bounds_ptr = reuse_cache.expert_bounds;
            src1_q8_1_ptr     = reuse_cache.src1_q8_1;
            reuse_cache.ready = false;
            ggml_cuda_mmq_ids_reuse_log("hit", device, src1, dst);
        } else {
            if (reuse_cache.ready) {
                reuse_cache.ready = false;
                ggml_cuda_mmq_ids_reuse_log("invalidate", device, src1, dst);
            }

            if (reuse_cacheable) {
                ggml_cuda_mmq_device_realloc(device, (void **) &reuse_cache.ids_src1, &reuse_cache.ids_src1_actual,
                    ne_get_rows * sizeof(int32_t));
                ggml_cuda_mmq_device_realloc(device, (void **) &reuse_cache.ids_dst, &reuse_cache.ids_dst_actual,
                    ne_get_rows * sizeof(int32_t));
                ggml_cuda_mmq_device_realloc(device, (void **) &reuse_cache.expert_bounds, &reuse_cache.expert_bounds_actual,
                    (ne02 + 1) * sizeof(int32_t));

                if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                    ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "before", "src1_q8_1_ids_reuse",
                        nbytes_src1_q8_1, reuse_cache.src1_q8_1_actual, src0, src1, ids, dst);
                }
                ggml_cuda_mmq_device_realloc(device, (void **) &reuse_cache.src1_q8_1, &reuse_cache.src1_q8_1_actual,
                    nbytes_src1_q8_1);
                if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                    ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "after", "src1_q8_1_ids_reuse",
                        nbytes_src1_q8_1, reuse_cache.src1_q8_1_actual, src0, src1, ids, dst);
                }

                ids_src1_ptr      = reuse_cache.ids_src1;
                ids_dst_ptr       = reuse_cache.ids_dst;
                expert_bounds_ptr = reuse_cache.expert_bounds;
                src1_q8_1_ptr     = reuse_cache.src1_q8_1;
            } else {
                ids_src1_local.alloc(ctx.pool(), ne_get_rows);
                ids_dst_local.alloc(ctx.pool(), ne_get_rows);
                expert_bounds_local.alloc(ctx.pool(), ne02 + 1);

                if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                    ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "before", "src1_q8_1_ids",
                        nbytes_src1_q8_1, 0, src0, src1, ids, dst);
                }
                src1_q8_1_local.alloc(ctx.pool(), nbytes_src1_q8_1);
                if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                    ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "after", "src1_q8_1_ids",
                        nbytes_src1_q8_1, src1_q8_1_local.actual_size, src0, src1, ids, dst);
                }

                ids_src1_ptr      = ids_src1_local.get();
                ids_dst_ptr       = ids_dst_local.get();
                expert_bounds_ptr = expert_bounds_local.get();
                src1_q8_1_ptr     = src1_q8_1_local.get();
            }

            const int32_t * ids_chunk = (const int32_t *) ((const char *) ids->data + token0*ids->nb[1]);

            ggml_cuda_launch_mm_ids_helper(ids_chunk, ids_src1_ptr, ids_dst_ptr, expert_bounds_ptr,
                ne02, ne12_chunk, n_expert_used, ne11, ids_si1, ids_sis1, stream);
            CUDA_CHECK(cudaGetLastError());

            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12_src1 = src1->nb[2] / ts_src1;
            const int64_t s13_src1 = src1->nb[3] / ts_src1;
            const float * src1_chunk_d = src1_d + token0*s12_src1;

            if (use_native_fp4) {
                quantize_mmq_fp4_cuda(src1_chunk_d, ids_src1_ptr, src1_q8_1_ptr, src0->type, ne10, s11, s12_src1, s13_src1,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            } else {
                quantize_mmq_q8_1_cuda(src1_chunk_d, ids_src1_ptr, src1_q8_1_ptr, src0->type, ne10, s11, s12_src1, s13_src1,
                                       ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
            CUDA_CHECK(cudaGetLastError());

            if (reuse_cacheable) {
                reuse_cache.ready = true;
                reuse_cache.device = device;
                reuse_cache.stream = stream;
                reuse_cache.src1 = src1;
                reuse_cache.ids = ids;
                reuse_cache.type_x = src0->type;
                reuse_cache.token0 = token0;
                reuse_cache.ne10 = ne10;
                reuse_cache.ne10_padded = ne10_padded;
                reuse_cache.ne11 = ne11;
                reuse_cache.ne12 = ne12;
                reuse_cache.ne12_chunk = ne12_chunk;
                reuse_cache.ne_get_rows = ne_get_rows;
                reuse_cache.ne02 = ne02;
                reuse_cache.n_expert_used = n_expert_used;
                reuse_cache.ids_si1 = ids_si1;
                reuse_cache.ids_sis1 = ids_sis1;
                ggml_cuda_mmq_ids_reuse_log("fill", device, src1, dst);
            }
        }

        static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
        const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                             ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12_chunk*s12;

        // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1_ptr, ids_dst_ptr, expert_bounds_ptr, dst_d + token0*s2,
            ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
            ne02, ne02, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k && !ids_chunk_no_stream_k, ne12_chunk, ids_chunk_force_x};

        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    }
}

void ggml_cuda_mul_mat_q_precompacted_ids(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const int32_t * ids_dst, const int32_t * expert_bounds, ggml_tensor * dst,
        const ggml_tensor * src1_key, const ggml_tensor * ids_key,
        const int64_t expert_low, const int64_t expert_high) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids_dst != nullptr);
    GGML_ASSERT(expert_bounds != nullptr);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);
    GGML_ASSERT(ne12 == 1 && ne13 == 1);
    GGML_ASSERT(ne2  == 1 && ne3  == 1);

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const bool no_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_NO_STREAM_K");
    const bool use_stream_k = !no_stream_k &&
                            ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc));

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t compact_rows = ne11;
    const int64_t local_experts = ne02;

    const int device = ggml_cuda_get_device();
    static int alloc_debug_count = 0;

    const size_t nbytes_src1_q8_1 = compact_rows * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    const bool reuse_cacheable = src1_key && ids_key && ggml_cuda_mmq_precomp_reuse_cacheable(cc);
    ggml_cuda_mmq_precomp_q8_reuse_cache & reuse_cache = g_mmq_precomp_q8_reuse_cache[device];
    if (reuse_cacheable && reuse_cache.stream != nullptr && reuse_cache.stream != stream) {
        reuse_cache.ready = false;
    }

    char * src1_q8_1_ptr = nullptr;
    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());
    const bool reuse_hit = reuse_cacheable && ggml_cuda_mmq_precomp_reuse_matches(
        reuse_cache, device, stream, src1_key, ids_key, src0->type, expert_low, expert_high, ne10, ne10_padded, compact_rows);

    if (reuse_hit) {
        src1_q8_1_ptr = reuse_cache.src1_q8_1;
        reuse_cache.ready = false;
        ggml_cuda_mmq_precomp_reuse_log("hit", device, src1_key, ids_key);
    } else {
        if (reuse_cacheable) {
            ggml_cuda_mmq_device_realloc(device, (void **) &reuse_cache.src1_q8_1, &reuse_cache.src1_q8_1_actual, nbytes_src1_q8_1);
            src1_q8_1_ptr = reuse_cache.src1_q8_1;
        } else {
            if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "before", "src1_q8_1_precompacted_ids",
                    nbytes_src1_q8_1, 0, src0, src1, nullptr, dst);
            }
            src1_q8_1_local.alloc(ctx.pool(), nbytes_src1_q8_1);
            if (ggml_cuda_mul_mat_alloc_debug_should_log(device, alloc_debug_count)) {
                ggml_cuda_mul_mat_alloc_debug_log(__func__, device, "after", "src1_q8_1_precompacted_ids",
                    nbytes_src1_q8_1, src1_q8_1_local.actual_size, src0, src1, nullptr, dst);
            }
            src1_q8_1_ptr = src1_q8_1_local.get();
        }

        const float * src1_d = (const float *) src1->data;
        const int64_t s11_src1 = src1->nb[1] / ts_src1;
        const int64_t s12_src1 = src1->nb[2] / ts_src1;
        const int64_t s13_src1 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1_ptr, src0->type, ne10,
                s11_src1, s12_src1, s13_src1, ne10_padded, compact_rows, 1, 1, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1_ptr, src0->type, ne10,
                s11_src1, s12_src1, s13_src1, ne10_padded, compact_rows, 1, 1, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        if (reuse_cacheable) {
            reuse_cache.ready = true;
            reuse_cache.device = device;
            reuse_cache.stream = stream;
            reuse_cache.src1_key = src1_key;
            reuse_cache.ids_key = ids_key;
            reuse_cache.type_x = src0->type;
            reuse_cache.expert_low = expert_low;
            reuse_cache.expert_high = expert_high;
            reuse_cache.ne10 = ne10;
            reuse_cache.ne10_padded = ne10_padded;
            reuse_cache.compact_rows = compact_rows;
            ggml_cuda_mmq_precomp_reuse_log("fill", device, src1_key, ids_key);
        }
    }

    const char * src0_d = (const char *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  = dst->nb[3] / ts_dst;

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? compact_rows * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         compact_rows * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1_ptr, ids_dst, expert_bounds, dst_d,
        ne00, ne01, compact_rows, s01, compact_rows, s1,
        local_experts, local_experts, s02, s12, s2,
        ne03, 1, s03, s12, s3,
        use_stream_k, compact_rows, 0};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool no_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_mmq_env_enabled("GGML_CUDA_MMQ_NO_STREAM_K");
    const bool use_stream_k = !no_stream_k &&
                            ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols, 0};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
