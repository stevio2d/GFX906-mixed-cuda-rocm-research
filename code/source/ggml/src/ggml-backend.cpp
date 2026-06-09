// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

static bool ggml_backend_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && value[0] != '\0' && value[0] != '0' && value[0] != 'f' && value[0] != 'F';
}

static size_t ggml_backend_env_size(const char * name, size_t default_value) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    char * end = NULL;
    double parsed = strtod(value, &end);
    if (end == value || parsed <= 0.0) {
        return default_value;
    }

    size_t scale = 1;
    if (*end == 'k' || *end == 'K') {
        scale = 1024;
    } else if (*end == 'm' || *end == 'M') {
        scale = 1024 * 1024;
    } else if (*end == 'g' || *end == 'G') {
        scale = 1024ull * 1024ull * 1024ull;
    }

    return (size_t) (parsed * scale);
}

static int ggml_backend_env_int(const char * name, int default_value) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    char * end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > INT_MAX) {
        return default_value;
    }

    return (int) parsed;
}

static double ggml_backend_env_double(const char * name, double default_value) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    char * end = NULL;
    const double parsed = strtod(value, &end);
    if (end == value) {
        return default_value;
    }

    return parsed;
}

static void * ggml_backend_aligned_malloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void * ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void ggml_backend_aligned_free(void * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

typedef bool (*ggml_backend_register_host_buffer_fn)(void * buffer, size_t size);
typedef void (*ggml_backend_unregister_host_buffer_fn)(void * buffer);
typedef bool (*ggml_backend_get_tensor_sync_fn)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
typedef bool (*ggml_backend_set_tensor_sync_fn)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

static bool ggml_backend_mixed_reg_supported(ggml_backend_reg_t reg) {
    if (reg == NULL) {
        return false;
    }

    const char * name = ggml_backend_reg_name(reg);
    return strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0;
}

static ggml_backend_buffer_t ggml_backend_tensor_buffer(const struct ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

static ggml_backend_reg_t ggml_backend_tensor_reg(const struct ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = ggml_backend_tensor_buffer(tensor);
    if (buffer == NULL) {
        return NULL;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    if (buft == NULL) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == NULL) {
        return NULL;
    }

    return ggml_backend_dev_backend_reg(dev);
}

static bool ggml_backend_mixed_register(
        ggml_backend_reg_t                         reg,
        void *                                     ptr,
        size_t                                     size,
        ggml_backend_unregister_host_buffer_fn *   unregister_fn) {
    *unregister_fn = NULL;

    ggml_backend_register_host_buffer_fn register_fn =
        (ggml_backend_register_host_buffer_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
    if (register_fn == NULL) {
        return false;
    }

    if (!register_fn(ptr, size)) {
        return false;
    }

    *unregister_fn =
        (ggml_backend_unregister_host_buffer_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
    return true;
}

struct ggml_backend_mixed_stage {
    void * ptr = NULL;
    size_t capacity = 0;
    ggml_backend_reg_t src_reg = NULL;
    ggml_backend_reg_t dst_reg = NULL;
    ggml_backend_unregister_host_buffer_fn unregister_src = NULL;
    ggml_backend_unregister_host_buffer_fn unregister_dst = NULL;

    ggml_backend_mixed_stage() = default;
    ggml_backend_mixed_stage(const ggml_backend_mixed_stage &) = delete;
    ggml_backend_mixed_stage & operator=(const ggml_backend_mixed_stage &) = delete;

    ~ggml_backend_mixed_stage() {
        reset();
    }

    bool matches(ggml_backend_reg_t src, ggml_backend_reg_t dst, size_t size) const {
        return ptr != NULL && src_reg == src && dst_reg == dst && capacity >= size;
    }

    bool init(ggml_backend_reg_t src, ggml_backend_reg_t dst, size_t size) {
        reset();

        ptr = ggml_backend_aligned_malloc(4096, size);
        if (ptr == NULL) {
            return false;
        }

        src_reg = src;
        dst_reg = dst;
        capacity = size;

        const bool registered_src = ggml_backend_mixed_register(src_reg, ptr, capacity, &unregister_src);
        const bool registered_dst = ggml_backend_mixed_register(dst_reg, ptr, capacity, &unregister_dst);

        if (!registered_src || !registered_dst) {
            reset();
            return false;
        }

        return true;
    }

    void reset() {
        if (ptr != NULL) {
            if (unregister_dst != NULL) {
                unregister_dst(ptr);
            }
            if (unregister_src != NULL) {
                unregister_src(ptr);
            }
            ggml_backend_aligned_free(ptr);
        }

        ptr = NULL;
        capacity = 0;
        src_reg = NULL;
        dst_reg = NULL;
        unregister_src = NULL;
        unregister_dst = NULL;
    }
};

static bool ggml_backend_mixed_copy_cache_enabled() {
    const char * value = getenv("GGML_MIXED_VENDOR_COPY_CACHE");
    return value == NULL || (value[0] != '\0' && value[0] != '0' && value[0] != 'f' && value[0] != 'F');
}

static ggml_backend_mixed_stage * ggml_backend_mixed_stage_get(
        ggml_backend_reg_t src_reg,
        ggml_backend_reg_t dst_reg,
        size_t             host_size) {
    static constexpr int max_stages = 4;
    thread_local ggml_backend_mixed_stage stages[max_stages];

    int n_stages = ggml_backend_env_int("GGML_MIXED_VENDOR_COPY_CACHE_ENTRIES", max_stages);
    n_stages = std::max(1, std::min(n_stages, max_stages));

    for (int i = 0; i < n_stages; ++i) {
        if (stages[i].matches(src_reg, dst_reg, host_size)) {
            return &stages[i];
        }
    }

    for (int i = 0; i < n_stages; ++i) {
        if (stages[i].ptr == NULL) {
            return stages[i].init(src_reg, dst_reg, host_size) ? &stages[i] : NULL;
        }
    }

    return stages[0].init(src_reg, dst_reg, host_size) ? &stages[0] : NULL;
}

static bool ggml_backend_tensor_copy_mixed_vendor(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!ggml_backend_env_enabled("GGML_MIXED_VENDOR_COPY")) {
        return false;
    }

    const size_t nbytes = ggml_nbytes(src);
    const size_t min_bytes = ggml_backend_env_size("GGML_MIXED_VENDOR_COPY_MIN_BYTES", 4ull * 1024ull * 1024ull);
    if (nbytes < min_bytes || nbytes == 0) {
        return false;
    }

    ggml_backend_buffer_t src_buffer = ggml_backend_tensor_buffer(src);
    ggml_backend_buffer_t dst_buffer = ggml_backend_tensor_buffer(dst);
    if (src_buffer == NULL || dst_buffer == NULL ||
            ggml_backend_buffer_is_host(src_buffer) || ggml_backend_buffer_is_host(dst_buffer)) {
        return false;
    }

    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    ggml_backend_reg_t src_reg = ggml_backend_tensor_reg(src);
    ggml_backend_reg_t dst_reg = ggml_backend_tensor_reg(dst);
    if (src_reg == NULL || dst_reg == NULL || src_reg == dst_reg) {
        return false;
    }

    if (!ggml_backend_mixed_reg_supported(src_reg) || !ggml_backend_mixed_reg_supported(dst_reg)) {
        return false;
    }

    ggml_backend_get_tensor_sync_fn get_tensor_sync = NULL;
    ggml_backend_set_tensor_sync_fn set_tensor_sync = NULL;
    if (backend_src != NULL && backend_dst != NULL) {
        ggml_backend_reg_t backend_src_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_src));
        ggml_backend_reg_t backend_dst_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_dst));
        if (backend_src_reg == src_reg && backend_dst_reg == dst_reg) {
            get_tensor_sync = (ggml_backend_get_tensor_sync_fn)
                ggml_backend_reg_get_proc_address(src_reg, "ggml_backend_get_tensor_sync");
            set_tensor_sync = (ggml_backend_set_tensor_sync_fn)
                ggml_backend_reg_get_proc_address(dst_reg, "ggml_backend_set_tensor_sync");
        }
    }

    size_t chunk_size = ggml_backend_env_size("GGML_MIXED_VENDOR_COPY_CHUNK_BYTES", 8ull * 1024ull * 1024ull);
    if (chunk_size == 0) {
        return false;
    }
    chunk_size = std::min(chunk_size, nbytes);

    int n_slots_requested = ggml_backend_env_int("GGML_MIXED_VENDOR_COPY_BUFFERS", 3);
    n_slots_requested = std::max(1, std::min(n_slots_requested, 16));

    const size_t n_chunks = (nbytes + chunk_size - 1) / chunk_size;
    const int n_slots = (int) std::min<size_t>((size_t) n_slots_requested, n_chunks);
    if (chunk_size > SIZE_MAX / (size_t) n_slots) {
        return false;
    }

    const size_t host_size = chunk_size * (size_t) n_slots;
    ggml_backend_mixed_stage local_stage;
    ggml_backend_mixed_stage * stage = NULL;
    if (ggml_backend_mixed_copy_cache_enabled()) {
        stage = ggml_backend_mixed_stage_get(src_reg, dst_reg, host_size);
    } else if (local_stage.init(src_reg, dst_reg, host_size)) {
        stage = &local_stage;
    }
    if (stage == NULL || stage->ptr == NULL) {
        return false;
    }

    void * host_ptr = stage->ptr;

    struct copy_slot {
        uint8_t * ptr;
        size_t offset;
        size_t size;
        bool full;
    };

    std::vector<copy_slot> slots(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        slots[i] = { (uint8_t *) host_ptr + (size_t) i * chunk_size, 0, 0, false };
    }

    std::mutex mutex;
    std::condition_variable cv_empty;
    std::condition_variable cv_full;

    const int64_t t_start_us = ggml_time_us();

    std::thread producer([&]() {
        for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
            copy_slot & slot = slots[chunk % (size_t) n_slots];
            const size_t offset = chunk * chunk_size;
            const size_t size = std::min(chunk_size, nbytes - offset);

            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_empty.wait(lock, [&]() { return !slot.full; });
                slot.offset = offset;
                slot.size = size;
            }

            if (get_tensor_sync != NULL) {
                get_tensor_sync(backend_src, src, slot.ptr, offset, size);
            } else {
                ggml_backend_tensor_get(src, slot.ptr, offset, size);
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                slot.full = true;
            }
            cv_full.notify_one();
        }
    });

    std::thread consumer([&]() {
        for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
            copy_slot & slot = slots[chunk % (size_t) n_slots];
            size_t offset;
            size_t size;

            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_full.wait(lock, [&]() { return slot.full; });
                offset = slot.offset;
                size = slot.size;
            }

            if (set_tensor_sync != NULL) {
                set_tensor_sync(backend_dst, dst, slot.ptr, offset, size);
            } else {
                ggml_backend_tensor_set(dst, slot.ptr, offset, size);
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                slot.full = false;
            }
            cv_empty.notify_one();
        }
    });

    producer.join();
    consumer.join();

    if (ggml_backend_env_enabled("GGML_MIXED_VENDOR_COPY_LOG")) {
        const int64_t t_elapsed_us = ggml_time_us() - t_start_us;
        const double seconds = t_elapsed_us > 0 ? t_elapsed_us / 1000000.0 : 0.0;
        const double gib = nbytes / (1024.0 * 1024.0 * 1024.0);
        const double gib_s = seconds > 0.0 ? gib / seconds : 0.0;
        GGML_LOG_INFO("%s: %s -> %s %.2f MiB chunk=%.2f MiB buffers=%d copy=%s bandwidth=%.2f GiB/s\n",
            __func__, ggml_backend_reg_name(src_reg), ggml_backend_reg_name(dst_reg),
            nbytes / (1024.0 * 1024.0), chunk_size / (1024.0 * 1024.0), n_slots,
            get_tensor_sync != NULL && set_tensor_sync != NULL ? "backend-sync" : "buffer",
            gib_s);
    }

    return true;
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        if (ggml_backend_tensor_copy_mixed_vendor(NULL, NULL, src, dst)) {
            return;
        }
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    if (ggml_backend_tensor_copy_mixed_vendor(backend_src, backend_dst, src, dst)) {
        return;
    }
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_moe_profile_stats {
    struct top_entry {
        int      expert_id = -1;
        uint64_t count     = 0;
    };

    struct node_record {
        int layer = -1;
        std::string backend_name;
        std::string weights_name;
        std::string ids_name;
        uint64_t full_weight_bytes = 0;
        uint64_t active_weight_bytes_est = 0;
        uint64_t expert_selections = 0;
        uint64_t unique_experts = 0;
        int64_t n_expert = 0;
        int64_t n_topk = 0;
        int64_t n_tokens = 0;
        std::vector<uint64_t> shard_selections;
        std::vector<uint64_t> shard_unique;
        std::vector<top_entry> top_experts;
    };

    struct layer_record {
        int layer = -1;
        int64_t n_expert = 0;
        uint64_t op_count = 0;
        uint64_t expert_selections = 0;
        uint64_t unique_experts_total = 0;
        uint64_t active_weight_bytes_est = 0;
        uint64_t full_weight_bytes = 0;
        uint64_t max_unique_experts = 0;
        uint64_t max_topk = 0;
        uint64_t max_tokens = 0;
        std::vector<uint64_t> expert_hits;
    };

    uint64_t eval_count;
    uint64_t op_count;
    uint64_t expert_selections;
    uint64_t unique_experts_total;
    uint64_t active_weight_bytes_est;
    uint64_t full_weight_bytes;
    uint64_t max_unique_experts;
    uint64_t max_experts;
    uint64_t max_topk;
    uint64_t max_tokens;
    int shard_count;
    uint64_t shard_max_selections;
    uint64_t shard_max_unique;
    std::vector<uint64_t> shard_selections;
    std::vector<uint64_t> shard_unique_total;
    std::vector<node_record> node_records;
    std::map<int, layer_record> layer_records;
    const ggml_tensor * prev_ids_tensor = nullptr;
    int64_t prev_ids_n_expert = -1;
    uint64_t prev_unique_experts = 0;
    std::vector<uint64_t> prev_expert_hits;
    std::string json_path;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    bool moe_profile;
    std::string moe_profile_json;
    uint64_t moe_profile_eval_count;

    int debug;

    bool profile_splits;
    bool profile_nodes;
    bool profile_copy;
    bool profile_copy_detail;
    int  profile_copy_max;
    int  profile_copy_count;
    char profile_label[64];

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

static uint64_t ggml_backend_sched_bitset_count(const ggml_bitset_t * bitset, int64_t n_bits) {
    uint64_t result = 0;
    for (int64_t i = 0; i < n_bits; ++i) {
        if (ggml_bitset_get(bitset, i)) {
            ++result;
        }
    }
    return result;
}

static int ggml_backend_sched_moe_profile_parse_layer(const char * tensor_name) {
    if (tensor_name == nullptr || strncmp(tensor_name, "blk.", 4) != 0) {
        return -1;
    }

    int layer = 0;
    bool saw_digit = false;
    for (const char * p = tensor_name + 4; *p; ++p) {
        if (*p >= '0' && *p <= '9') {
            saw_digit = true;
            layer = layer * 10 + (*p - '0');
            continue;
        }
        if (*p == '.' && saw_digit) {
            return layer;
        }
        break;
    }

    return -1;
}

static std::vector<ggml_backend_sched_moe_profile_stats::top_entry> ggml_backend_sched_moe_profile_top_experts(
        const std::vector<uint64_t> & expert_hits,
        size_t max_entries) {
    std::vector<ggml_backend_sched_moe_profile_stats::top_entry> top;
    top.reserve(expert_hits.size());

    for (size_t i = 0; i < expert_hits.size(); ++i) {
        if (expert_hits[i] == 0) {
            continue;
        }
        top.push_back({(int) i, expert_hits[i]});
    }

    std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) {
        if (a.count != b.count) {
            return a.count > b.count;
        }
        return a.expert_id < b.expert_id;
    });

    if (top.size() > max_entries) {
        top.resize(max_entries);
    }

    return top;
}

static std::string ggml_backend_sched_moe_profile_json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) ch);
                    escaped += buf;
                } else {
                    escaped += (char) ch;
                }
                break;
        }
    }

    return escaped;
}

static void ggml_backend_sched_moe_profile_node(
        ggml_backend_sched_t sched,
        ggml_backend_t       split_backend,
        ggml_tensor        * node,
        ggml_backend_sched_moe_profile_stats & stats) {
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr || node->src[2] == nullptr) {
        return;
    }

    ggml_tensor * weights    = node->src[0];
    ggml_tensor * ids_tensor = node->src[2];

    if (ids_tensor->type != GGML_TYPE_I32 || weights->ne[2] <= 0) {
        return;
    }

    const int64_t n_expert    = weights->ne[2];
    const size_t  expert_size = weights->nb[2];
    const int64_t n_topk      = ids_tensor->ne[0];
    const int64_t n_tokens    = ids_tensor->ne[1];

    if (n_expert <= 0 || n_topk <= 0 || n_tokens <= 0 || expert_size == 0) {
        return;
    }

    ggml_backend_t ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_tensor);
    if (ids_backend == nullptr) {
        ids_backend = split_backend;
    }

    ggml_backend_sched_moe_profile_stats::node_record node_record;
    std::vector<uint64_t> expert_hits;
    uint64_t unique_experts = 0;

    if (stats.prev_ids_tensor == ids_tensor && stats.prev_ids_n_expert == n_expert) {
        expert_hits = stats.prev_expert_hits;
        unique_experts = stats.prev_unique_experts;
    } else {
        std::vector<int32_t> ids(ggml_nbytes(ids_tensor) / sizeof(int32_t));
        std::vector<ggml_bitset_t> used_ids(ggml_bitset_size(n_expert));
        expert_hits.resize((size_t) n_expert);

        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
        ggml_backend_synchronize(ids_backend);

        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; ++i0) {
                const int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                ggml_bitset_set(used_ids.data(), id);
                expert_hits[(size_t) id]++;
            }
        }

        unique_experts = ggml_backend_sched_bitset_count(used_ids.data(), n_expert);
        stats.prev_ids_tensor = ids_tensor;
        stats.prev_ids_n_expert = n_expert;
        stats.prev_unique_experts = unique_experts;
        stats.prev_expert_hits = expert_hits;
    }

    std::vector<uint64_t> node_shard_selections;
    std::vector<uint64_t> node_shard_unique;
    if (stats.shard_count > 0) {
        node_shard_selections.resize(stats.shard_count);
        node_shard_unique.resize(stats.shard_count);
        for (int64_t ie = 0; ie < n_expert; ++ie) {
            const uint64_t hit_count = expert_hits[(size_t) ie];
            if (hit_count == 0) {
                continue;
            }
            const int shard = std::min<int64_t>((int64_t) stats.shard_count - 1, (ie * stats.shard_count) / n_expert);
            node_shard_selections[shard] += hit_count;
            node_shard_unique[shard]++;
        }
    }

    stats.op_count++;
    stats.expert_selections       += (uint64_t) n_topk * (uint64_t) n_tokens;
    stats.unique_experts_total    += unique_experts;
    stats.active_weight_bytes_est += unique_experts * (uint64_t) expert_size;
    stats.full_weight_bytes       += (uint64_t) ggml_nbytes(weights);
    stats.max_unique_experts       = std::max(stats.max_unique_experts, unique_experts);
    stats.max_experts              = std::max(stats.max_experts, (uint64_t) n_expert);
    stats.max_topk                 = std::max(stats.max_topk, (uint64_t) n_topk);
    stats.max_tokens               = std::max(stats.max_tokens, (uint64_t) n_tokens);

    if (stats.shard_count > 0) {
        for (int shard = 0; shard < stats.shard_count; ++shard) {
            stats.shard_selections[shard] += node_shard_selections[shard];
            stats.shard_unique_total[shard] += node_shard_unique[shard];
            stats.shard_max_selections = std::max(stats.shard_max_selections, node_shard_selections[shard]);
            stats.shard_max_unique     = std::max(stats.shard_max_unique,     node_shard_unique[shard]);
        }

        if (!stats.json_path.empty()) {
            node_record.shard_selections = std::move(node_shard_selections);
            node_record.shard_unique = std::move(node_shard_unique);
        }
    }

    if (!stats.json_path.empty()) {
        node_record.layer = ggml_backend_sched_moe_profile_parse_layer(weights->name);
        node_record.backend_name = split_backend != nullptr ? ggml_backend_name(split_backend) : "";
        node_record.weights_name = weights->name;
        node_record.ids_name = ids_tensor->name;
        node_record.full_weight_bytes = ggml_nbytes(weights);
        node_record.active_weight_bytes_est = unique_experts * (uint64_t) expert_size;
        node_record.expert_selections = (uint64_t) n_topk * (uint64_t) n_tokens;
        node_record.unique_experts = unique_experts;
        node_record.n_expert = n_expert;
        node_record.n_topk = n_topk;
        node_record.n_tokens = n_tokens;
        node_record.top_experts = ggml_backend_sched_moe_profile_top_experts(expert_hits, 8);
        stats.node_records.push_back(std::move(node_record));

        const int layer = stats.node_records.back().layer;
        if (layer >= 0) {
            auto & layer_record = stats.layer_records[layer];
            if (layer_record.layer < 0) {
                layer_record.layer = layer;
                layer_record.n_expert = n_expert;
                layer_record.expert_hits.resize((size_t) n_expert);
            }

            if (layer_record.expert_hits.size() < expert_hits.size()) {
                layer_record.expert_hits.resize(expert_hits.size());
            }

            layer_record.op_count++;
            layer_record.expert_selections += (uint64_t) n_topk * (uint64_t) n_tokens;
            layer_record.unique_experts_total += unique_experts;
            layer_record.active_weight_bytes_est += unique_experts * (uint64_t) expert_size;
            layer_record.full_weight_bytes += (uint64_t) ggml_nbytes(weights);
            layer_record.max_unique_experts = std::max(layer_record.max_unique_experts, unique_experts);
            layer_record.max_topk = std::max(layer_record.max_topk, (uint64_t) n_topk);
            layer_record.max_tokens = std::max(layer_record.max_tokens, (uint64_t) n_tokens);

            for (size_t i = 0; i < expert_hits.size(); ++i) {
                layer_record.expert_hits[i] += expert_hits[i];
            }
        }
    }
}

static void ggml_backend_sched_moe_profile_print(const ggml_backend_sched_moe_profile_stats & stats) {
    if (stats.op_count == 0) {
        GGML_LOG_WARN("moe_profile: eval=%" PRIu64 " mul_mat_id=0\n", stats.eval_count);
        return;
    }

    GGML_LOG_WARN(
        "moe_profile: eval=%" PRIu64
        " mul_mat_id=%" PRIu64
        " selections=%" PRIu64
        " unique_total=%" PRIu64
        " unique_avg=%.2f"
        " unique_max=%" PRIu64
        " max_experts=%" PRIu64
        " max_topk=%" PRIu64
        " max_tokens=%" PRIu64
        " active_weight_est=%.2f MiB"
        " full_weight=%.2f MiB\n",
        stats.eval_count,
        stats.op_count,
        stats.expert_selections,
        stats.unique_experts_total,
        (double) stats.unique_experts_total / (double) stats.op_count,
        stats.max_unique_experts,
        stats.max_experts,
        stats.max_topk,
        stats.max_tokens,
        (double) stats.active_weight_bytes_est / 1024.0 / 1024.0,
        (double) stats.full_weight_bytes / 1024.0 / 1024.0);

    if (stats.shard_count > 0 && !stats.shard_selections.empty()) {
        std::string selections;
        std::string unique_total;
        for (int shard = 0; shard < stats.shard_count; ++shard) {
            if (shard > 0) {
                selections += ",";
                unique_total += ",";
            }
            selections += std::to_string(stats.shard_selections[shard]);
            unique_total += std::to_string(stats.shard_unique_total[shard]);
        }

        const double avg_sel = (double) stats.expert_selections / (double) stats.shard_count;
        const uint64_t max_sel_total = *std::max_element(stats.shard_selections.begin(), stats.shard_selections.end());
        GGML_LOG_WARN(
            "moe_profile_shards: eval=%" PRIu64
            " shards=%d"
            " selections=%s"
            " unique_total=%s"
            " max_node_selections=%" PRIu64
            " max_node_unique=%" PRIu64
            " total_selection_imbalance=%.3f\n",
            stats.eval_count,
            stats.shard_count,
            selections.c_str(),
            unique_total.c_str(),
            stats.shard_max_selections,
            stats.shard_max_unique,
            avg_sel > 0.0 ? (double) max_sel_total / avg_sel : 0.0);
    }
}

static std::string ggml_backend_sched_moe_profile_json_path_for_eval(const std::string & base_path, uint64_t eval_count) {
    if (base_path.empty() || eval_count <= 1) {
        return base_path;
    }

    return base_path + ".eval" + std::to_string(eval_count) + ".json";
}

static void ggml_backend_sched_moe_profile_dump_json(const ggml_backend_sched_moe_profile_stats & stats) {
    if (stats.json_path.empty()) {
        return;
    }

    const std::string path = ggml_backend_sched_moe_profile_json_path_for_eval(stats.json_path, stats.eval_count);
    std::ofstream out(path);
    if (!out) {
        GGML_LOG_WARN("moe_profile_json: failed to open %s for writing\n", path.c_str());
        return;
    }

    auto write_u64_array = [&](const std::vector<uint64_t> & values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << values[i];
        }
        out << "]";
    };

    auto write_top_entries = [&](const std::vector<ggml_backend_sched_moe_profile_stats::top_entry> & top_entries) {
        out << "[";
        for (size_t i = 0; i < top_entries.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << "{\"expert_id\":" << top_entries[i].expert_id
                << ",\"count\":" << top_entries[i].count << "}";
        }
        out << "]";
    };

    out << "{\n";
    out << "  \"eval_count\": " << stats.eval_count << ",\n";
    out << "  \"op_count\": " << stats.op_count << ",\n";
    out << "  \"expert_selections\": " << stats.expert_selections << ",\n";
    out << "  \"unique_experts_total\": " << stats.unique_experts_total << ",\n";
    out << "  \"unique_experts_avg\": " << (stats.op_count ? (double) stats.unique_experts_total / (double) stats.op_count : 0.0) << ",\n";
    out << "  \"active_weight_bytes_est\": " << stats.active_weight_bytes_est << ",\n";
    out << "  \"full_weight_bytes\": " << stats.full_weight_bytes << ",\n";
    out << "  \"max_unique_experts\": " << stats.max_unique_experts << ",\n";
    out << "  \"max_experts\": " << stats.max_experts << ",\n";
    out << "  \"max_topk\": " << stats.max_topk << ",\n";
    out << "  \"max_tokens\": " << stats.max_tokens << ",\n";
    out << "  \"shard_count\": " << stats.shard_count << ",\n";
    out << "  \"shard_selections\": ";
    write_u64_array(stats.shard_selections);
    out << ",\n";
    out << "  \"shard_unique_total\": ";
    write_u64_array(stats.shard_unique_total);
    out << ",\n";
    out << "  \"layers\": [\n";

    size_t layer_index = 0;
    for (const auto & it : stats.layer_records) {
        const auto & layer = it.second;
        if (layer.layer < 0) {
            continue;
        }
        if (layer_index++ > 0) {
            out << ",\n";
        }
        out << "    {\"layer\":" << layer.layer
            << ",\"n_expert\":" << layer.n_expert
            << ",\"op_count\":" << layer.op_count
            << ",\"expert_selections\":" << layer.expert_selections
            << ",\"unique_experts_total\":" << layer.unique_experts_total
            << ",\"unique_experts_avg\":" << (layer.op_count ? (double) layer.unique_experts_total / (double) layer.op_count : 0.0)
            << ",\"active_weight_bytes_est\":" << layer.active_weight_bytes_est
            << ",\"full_weight_bytes\":" << layer.full_weight_bytes
            << ",\"max_unique_experts\":" << layer.max_unique_experts
            << ",\"max_topk\":" << layer.max_topk
            << ",\"max_tokens\":" << layer.max_tokens
            << ",\"top_experts\":";
        write_top_entries(ggml_backend_sched_moe_profile_top_experts(layer.expert_hits, 8));
        out << "}";
    }

    out << "\n  ],\n";
    out << "  \"nodes\": [\n";
    for (size_t i = 0; i < stats.node_records.size(); ++i) {
        const auto & node = stats.node_records[i];
        if (i > 0) {
            out << ",\n";
        }
        out << "    {\"layer\":" << node.layer
            << ",\"backend_name\":\"" << ggml_backend_sched_moe_profile_json_escape(node.backend_name) << "\""
            << ",\"weights_name\":\"" << ggml_backend_sched_moe_profile_json_escape(node.weights_name) << "\""
            << ",\"ids_name\":\"" << ggml_backend_sched_moe_profile_json_escape(node.ids_name) << "\""
            << ",\"full_weight_bytes\":" << node.full_weight_bytes
            << ",\"active_weight_bytes_est\":" << node.active_weight_bytes_est
            << ",\"expert_selections\":" << node.expert_selections
            << ",\"unique_experts\":" << node.unique_experts
            << ",\"n_expert\":" << node.n_expert
            << ",\"n_topk\":" << node.n_topk
            << ",\"n_tokens\":" << node.n_tokens
            << ",\"shard_selections\":";
        write_u64_array(node.shard_selections);
        out << ",\"shard_unique\":";
        write_u64_array(node.shard_unique);
        out << ",\"top_experts\":";
        write_top_entries(node.top_experts);
        out << "}";
    }
    out << "\n  ]\n";
    out << "}\n";

    GGML_LOG_WARN("moe_profile_json: wrote %s\n", path.c_str());
}

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static bool ggml_sched_starts_with(const char * str, const char * prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

static bool ggml_sched_profile_node_boundary(const char * name) {
    return
        ggml_sched_starts_with(name, "__fattn__-")     ||
        ggml_sched_starts_with(name, "kqv_out-")       ||
        ggml_sched_starts_with(name, "ffn_moe_out-")   ||
        ggml_sched_starts_with(name, "l_out-")         ||
        strcmp(name, "result_output") == 0;
}

static const char * ggml_sched_profile_node_kind(const char * name) {
    if (ggml_sched_starts_with(name, "__fattn__-") || ggml_sched_starts_with(name, "kqv_out-")) {
        return "attn";
    }

    if (ggml_sched_starts_with(name, "ffn_moe_out-")) {
        return "moe";
    }

    if (ggml_sched_starts_with(name, "l_out-")) {
        return "layer";
    }

    if (strcmp(name, "result_output") == 0) {
        return "output";
    }

    return "other";
}

static void ggml_sched_dump_moe_chunk(
        const char * label,
        int split_id,
        int n_splits,
        int chunk_id,
        ggml_backend_t backend,
        const struct ggml_backend_sched_split * split,
        int j0,
        int j1) {
    const int dump_chunks = ggml_backend_env_int("GGML_SCHED_MOE_DUMP_CHUNKS", 0);
    if (dump_chunks <= 0 || chunk_id > dump_chunks || j0 >= j1) {
        return;
    }

    const int dump_nodes = ggml_backend_env_int("GGML_SCHED_MOE_DUMP_NODES", 64);
    const int limit = dump_nodes > 0 ? std::min(j1, j0 + dump_nodes) : j1;

    GGML_LOG(
        "moe_chunk_dump: label=%s split=%d/%d chunk=%d backend=%s range=%d:%d nodes=%d dump_limit=%d\n",
        label && label[0] ? label : "-",
        split_id + 1,
        n_splits,
        chunk_id,
        backend ? ggml_backend_name(backend) : "NULL",
        split->i_start + j0,
        split->i_start + j1,
        j1 - j0,
        dump_nodes);

    for (int j = j0; j < limit; ++j) {
        ggml_tensor * node = split->graph.nodes[j];
        GGML_LOG(
            "moe_chunk_dump:   node=%d graph=%d name=%s op=%s src0=%s src1=%s src2=%s\n",
            j - j0,
            split->i_start + j,
            node ? node->name : "(null)",
            node ? ggml_op_name(node->op) : "(null)",
            node && node->src[0] ? node->src[0]->name : "(null)",
            node && node->src[1] ? node->src[1]->name : "(null)",
            node && node->src[2] ? node->src[2]->name : "(null)");
    }

    if (limit < j1) {
        GGML_LOG(
            "moe_chunk_dump:   truncated remaining=%d nodes after dump limit\n",
            j1 - limit);
    }
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static const char * ggml_backend_buffer_usage_name(enum ggml_backend_buffer_usage usage) {
    switch (usage) {
        case GGML_BACKEND_BUFFER_USAGE_ANY:
            return "any";
        case GGML_BACKEND_BUFFER_USAGE_WEIGHTS:
            return "weights";
        case GGML_BACKEND_BUFFER_USAGE_COMPUTE:
            return "compute";
        default:
            return "unknown";
    }
}

static void ggml_backend_sched_log_split_inputs(ggml_backend_sched_t sched) {
    const int debug_split_inputs = ggml_backend_env_int("GGML_SCHED_DEBUG_SPLIT_INPUTS", 0);
    if (debug_split_inputs <= 0) {
        return;
    }

    const int debug_input_limit = ggml_backend_env_int("GGML_SCHED_DEBUG_SPLIT_INPUT_LIMIT", 32);

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];

        size_t total_bytes = 0;
        size_t weight_bytes = 0;
        size_t user_input_bytes = 0;

        for (int input_idx = 0; input_idx < split->n_inputs; ++input_idx) {
            struct ggml_tensor * input = split->inputs[input_idx];
            const size_t nbytes = ggml_nbytes(input);
            total_bytes += nbytes;
            if (input->buffer && ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                weight_bytes += nbytes;
            }
            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                user_input_bytes += nbytes;
            }
        }

        GGML_LOG(
            "sched_split_inputs: split=%d/%d backend=%s range=%d:%d n_inputs=%d total=%zu weight=%zu user_input=%zu n_copies=%d cur_copy=%d\n",
            split_id + 1,
            sched->n_splits,
            ggml_backend_name(sched->backends[split->backend_id]),
            split->i_start,
            split->i_end,
            split->n_inputs,
            total_bytes,
            weight_bytes,
            user_input_bytes,
            sched->n_copies,
            sched->cur_copy);

        const int limit = debug_input_limit > 0 ? std::min(split->n_inputs, debug_input_limit) : split->n_inputs;
        for (int input_idx = 0; input_idx < limit; ++input_idx) {
            struct ggml_tensor * input = split->inputs[input_idx];
            const size_t id = hash_id(input);
            const int src_backend_id = sched->hv_tensor_backend_ids[id];
            ggml_backend_t src_backend = src_backend_id >= 0 ? sched->backends[src_backend_id] : nullptr;
            ggml_backend_buffer_t buf = input->view_src ? input->view_src->buffer : input->buffer;
            ggml_backend_buffer_type_t buft = buf ? buf->buft : nullptr;
            const char * buft_name = buft ? ggml_backend_buft_name(buft) : "(null)";
            const char * usage_name = buf ? ggml_backend_buffer_usage_name(ggml_backend_buffer_get_usage(buf)) : "unallocated";

            GGML_LOG(
                "sched_split_inputs:   input=%d name=%s bytes=%zu src_backend=%s usage=%s buft=%s flags[input=%d,output=%d]\n",
                input_idx,
                input->name,
                ggml_nbytes(input),
                src_backend ? ggml_backend_name(src_backend) : "NULL",
                usage_name,
                buft_name,
                (input->flags & GGML_TENSOR_FLAG_INPUT) ? 1 : 0,
                (input->flags & GGML_TENSOR_FLAG_OUTPUT) ? 1 : 0);
        }

        if (limit < split->n_inputs) {
            GGML_LOG("sched_split_inputs:   truncated remaining=%d inputs after limit\n", split->n_inputs - limit);
        }
    }
}

static bool ggml_backend_sched_input_copy_slots_on_rotate_only(void) {
    const char * env = getenv("GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY");
    return env != NULL && atoi(env) != 0 && getenv("LLAMA_PIPELINE_ROTATE_COPIES") == NULL;
}

static int ggml_backend_sched_input_copy_slot_count(ggml_backend_sched_t sched) {
    if (sched->n_copies <= 1) {
        return sched->n_copies;
    }

    return ggml_backend_sched_input_copy_slots_on_rotate_only() ? 1 : sched->n_copies;
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    ggml_backend_sched_log_split_inputs(sched);

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    const int input_copy_slots = ggml_backend_sched_input_copy_slot_count(sched);
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (input_copy_slots > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < input_copy_slots; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < input_copy_slots; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits_range(ggml_backend_sched_t sched, int split_start, int split_end) {
    GGML_ASSERT(sched);
    GGML_ASSERT(split_start >= 0);
    GGML_ASSERT(split_end < 0 || split_end >= split_start);

    split_start = std::min(split_start, sched->n_splits);
    split_end   = split_end < 0 ? sched->n_splits : std::min(split_end, sched->n_splits);

    struct ggml_backend_sched_split * splits = sched->splits;
    const bool profile_splits = sched->profile_splits && sched->callback_eval == nullptr;
    const bool profile_nodes  = sched->profile_nodes  && sched->callback_eval == nullptr;
    const bool profile_copy   = sched->profile_copy && sched->callback_eval == nullptr &&
        (sched->profile_copy_max <= 0 || sched->profile_copy_count < sched->profile_copy_max);
    const bool profile_copy_detail = profile_copy && sched->profile_copy_detail;
    const bool profile_copy_or_splits = profile_copy || profile_splits;
    const int profile_copy_call = profile_copy ? ++sched->profile_copy_count : sched->profile_copy_count;
    const bool copy_wait_source_event = ggml_backend_env_enabled("GGML_SCHED_COPY_WAIT_SOURCE_EVENT");
    const double moe_full_copy_ratio = ggml_backend_env_double("GGML_SCHED_MOE_FULL_COPY_RATIO", 2.0);

    if (profile_splits || profile_nodes) {
        GGML_LOG(
            "sched_profile: label=%s mode=%s n_splits=%d n_copies=%d cur_copy=%d\n",
            sched->profile_label[0] ? sched->profile_label : "-",
            profile_nodes ? "nodes" : "splits",
            sched->n_splits,
            sched->n_copies,
            sched->cur_copy);
    }

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
    uint64_t used_ids_count = 0;
    ggml_backend_sched_moe_profile_stats moe_profile_stats = {};
    if (sched->moe_profile) {
        moe_profile_stats.eval_count = ++sched->moe_profile_eval_count;
        moe_profile_stats.json_path = sched->moe_profile_json;
        if (const char * env = getenv("GGML_MOE_PROFILE_SHARDS")) {
            char * end = nullptr;
            const long value = strtol(env, &end, 10);
            if (end != env && value > 1 && value <= 16) {
                moe_profile_stats.shard_count = (int) value;
                moe_profile_stats.shard_selections.resize(moe_profile_stats.shard_count);
                moe_profile_stats.shard_unique_total.resize(moe_profile_stats.shard_count);
            }
        }
    }

    for (int split_id = split_start; split_id < split_end; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        size_t profile_input_bytes = 0;
        int profile_copy_async = 0;
        int profile_copy_mixed = 0;
        int profile_copy_sync  = 0;
        int profile_copy_user  = 0;
        int profile_copy_expert = 0;
        int profile_copy_expert_full = 0;
        int64_t profile_copy_wait_us = 0;
        int64_t profile_copy_dispatch_us = 0;

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            const int64_t profile_input_wait_before     = profile_copy_detail ? profile_copy_wait_us     : 0;
            const int64_t profile_input_dispatch_before = profile_copy_detail ? profile_copy_dispatch_us : 0;
            const int profile_input_async_before  = profile_copy_detail ? profile_copy_async  : 0;
            const int profile_input_mixed_before  = profile_copy_detail ? profile_copy_mixed  : 0;
            const int profile_input_sync_before   = profile_copy_detail ? profile_copy_sync   : 0;
            const int profile_input_user_before   = profile_copy_detail ? profile_copy_user   : 0;
            const int profile_input_expert_before = profile_copy_detail ? profile_copy_expert : 0;
            const int profile_input_expert_full_before = profile_copy_detail ? profile_copy_expert_full : 0;

            if (profile_copy_or_splits) {
                profile_input_bytes += ggml_nbytes(input);
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                int64_t profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                if (profile_copy_or_splits) {
                    profile_copy_wait_us += ggml_time_us() - profile_t_start_us;
                    profile_t_start_us = ggml_time_us();
                }
                ggml_backend_tensor_copy(input, input_cpy);
                if (profile_copy_or_splits) {
                    profile_copy_dispatch_us += ggml_time_us() - profile_t_start_us;
                    profile_copy_user++;
                    profile_copy_sync++;
                }
            } else {
                // wait for the split backend to finish using the input before overwriting it
                int64_t profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                if (profile_copy_or_splits) {
                    profile_copy_wait_us += ggml_time_us() - profile_t_start_us;
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                bool handled_copy = false;
                bool expert_full_copy = false;
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                    ggml_backend_synchronize(input_backend);
                    if (profile_copy_or_splits) {
                        profile_copy_wait_us += ggml_time_us() - profile_t_start_us;
                    }

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);
                        if (profile_copy_or_splits) {
                            profile_copy_wait_us += ggml_time_us() - profile_t_start_us;
                        }

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }
                        used_ids_count = ggml_backend_sched_bitset_count(used_ids.data(), n_expert);

                        prev_ids_tensor = ids_tensor;
                    }

                    const bool use_full_copy = moe_full_copy_ratio > 0.0 &&
                        moe_full_copy_ratio <= 1.0 &&
                        n_expert > 0 &&
                        (double) used_ids_count / (double) n_expert >= moe_full_copy_ratio;
                    expert_full_copy = use_full_copy;

                    if (!use_full_copy) {
                        // group consecutive experts and copy them together
                        auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                            const size_t expert_offset = first_id * expert_size;
                            const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                            const size_t padding = std::min<size_t>(expert_size, 512);
                            const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                            const int64_t copy_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                            ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                (const uint8_t *)input->data + expert_offset, expert_offset,
                                // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                                // this is necessary for MMQ in the CUDA backend
                                expert_size_copy + padding_end);
                            if (profile_copy_or_splits) {
                                profile_copy_dispatch_us += ggml_time_us() - copy_start_us;
                            }
                        };

                        int id = 0;
                        while (!ggml_bitset_get(used_ids.data(), id)) {
                            id++;
                        }
                        int32_t first_id = id;
                        int32_t last_id = first_id;

                        for (++id; id < n_expert; ++id) {
                            if (!ggml_bitset_get(used_ids.data(), id)) {
                                continue;
                            }

                            if (id == last_id + 1) {
                                last_id = id;
                                continue;
                            }

                            copy_experts(first_id, last_id);

                            first_id = id;
                            last_id = id;
                        }
                        copy_experts(first_id, last_id);
                        handled_copy = true;
                        if (profile_copy_or_splits) {
                            profile_copy_expert++;
                        }
                    }
                }

                if (!handled_copy) {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    bool async_ok = false;
                    profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                    if (split_backend->iface.cpy_tensor_async) {
                        async_ok = split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy);
                    }
                    if (profile_copy_or_splits) {
                        profile_copy_dispatch_us += ggml_time_us() - profile_t_start_us;
                    }
                    if (async_ok) {
                        if (profile_copy_or_splits) {
                            profile_copy_async++;
                        }
                    } else {
                        profile_t_start_us = profile_copy_or_splits ? ggml_time_us() : 0;
                        if (copy_wait_source_event) {
                            const int input_backend_id = ggml_backend_sched_backend_id(sched, input_backend);
                            if (input_backend_id >= 0 && sched->events[input_backend_id][sched->cur_copy] != NULL) {
                                ggml_backend_event_synchronize(sched->events[input_backend_id][sched->cur_copy]);
                            } else {
                                ggml_backend_synchronize(input_backend);
                            }
                        } else {
                            ggml_backend_synchronize(input_backend);
                        }
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        if (profile_copy_or_splits) {
                            profile_copy_wait_us += ggml_time_us() - profile_t_start_us;
                            profile_t_start_us = ggml_time_us();
                        }
                        const bool mixed_ok = ggml_backend_tensor_copy_mixed_vendor(input_backend, split_backend, input, input_cpy);
                        if (!mixed_ok) {
                            ggml_backend_tensor_copy(input, input_cpy);
                        }
                        if (profile_copy_or_splits) {
                            profile_copy_dispatch_us += ggml_time_us() - profile_t_start_us;
                            if (mixed_ok) {
                                profile_copy_mixed++;
                            } else {
                                profile_copy_sync++;
                            }
                        }
                    }
                    if (profile_copy_or_splits && expert_full_copy) {
                        profile_copy_expert_full++;
                    }
                }
            }

            if (profile_copy_detail) {
                const int delta_async  = profile_copy_async  - profile_input_async_before;
                const int delta_mixed  = profile_copy_mixed  - profile_input_mixed_before;
                const int delta_sync   = profile_copy_sync   - profile_input_sync_before;
                const int delta_user   = profile_copy_user   - profile_input_user_before;
                const int delta_expert = profile_copy_expert - profile_input_expert_before;
                const int delta_expert_full = profile_copy_expert_full - profile_input_expert_full_before;

                const char * copy_kind =
                    delta_user   > 0 ? "user"   :
                    delta_expert_full > 0 ? "expert_full" :
                    delta_expert > 0 ? "expert" :
                    delta_async  > 0 ? "async"  :
                    delta_mixed  > 0 ? "mixed"  :
                    delta_sync   > 0 ? "sync"   : "none";

                GGML_LOG(
                    "sched_copy_detail: label=%s call=%d split=%d/%d input=%d/%d src_backend=%s dst_backend=%s bytes=%.2f MiB wait_ms=%.3f dispatch_ms=%.3f kind=%s src=%s dst=%s flags=0x%x\n",
                    sched->profile_label[0] ? sched->profile_label : "-",
                    profile_copy_call,
                    split_id + 1,
                    sched->n_splits,
                    input_id + 1,
                    split->n_inputs,
                    input_backend ? ggml_backend_name(input_backend) : "NULL",
                    ggml_backend_name(split_backend),
                    ggml_nbytes(input) / 1048576.0,
                    (profile_copy_wait_us     - profile_input_wait_before)     / 1000.0,
                    (profile_copy_dispatch_us - profile_input_dispatch_before) / 1000.0,
                    copy_kind,
                    input->name,
                    input_cpy ? input_cpy->name : "NULL",
                    input->flags);
            }
        }

        if (profile_copy_or_splits && split->n_inputs > 0) {
            GGML_LOG(
                "%s: label=%s call=%d split=%d/%d backend=%s copy_inputs=%d copy_bytes=%.2f MiB copy_wait_ms=%.3f copy_dispatch_ms=%.3f async=%d mixed=%d sync=%d user=%d expert=%d\n",
                profile_copy ? "sched_copy_profile" : "sched_profile",
                sched->profile_label[0] ? sched->profile_label : "-",
                profile_copy_call,
                split_id + 1,
                sched->n_splits,
                ggml_backend_name(split_backend),
                split->n_inputs,
                profile_input_bytes / 1048576.0,
                profile_copy_wait_us / 1000.0,
                profile_copy_dispatch_us / 1000.0,
                profile_copy_async,
                profile_copy_mixed,
                profile_copy_sync,
                profile_copy_user,
                profile_copy_expert + profile_copy_expert_full);
        }

        if (!sched->callback_eval) {
            auto profile_compute_view = [&](int j0, int j1, int chunk_id, const char * kind, bool log_profile) -> enum ggml_status {
                if (j0 >= j1) {
                    return GGML_STATUS_SUCCESS;
                }

                ggml_backend_synchronize(split_backend);
                const int64_t t_start_us = ggml_time_us();

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1);
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                ggml_backend_synchronize(split_backend);

                if (log_profile) {
                    const char * first = gv.n_nodes > 0 ? gv.nodes[0]->name : "";
                    const char * last  = gv.n_nodes > 0 ? gv.nodes[gv.n_nodes - 1]->name : "";
                    const char * label = sched->profile_label[0] ? sched->profile_label : "-";

                    GGML_LOG(
                        "sched_node_profile: label=%s split=%d/%d chunk=%d backend=%s kind=%s nodes=%d range=%d:%d compute_ms=%.3f first=%s last=%s\n",
                        label,
                        split_id + 1,
                        sched->n_splits,
                        chunk_id,
                        ggml_backend_name(split_backend),
                        kind,
                        gv.n_nodes,
                        split->i_start + j0,
                        split->i_start + j1,
                        (ggml_time_us() - t_start_us) / 1000.0,
                        first,
                        last);

                    if (strcmp(kind, "moe") == 0) {
                        ggml_sched_dump_moe_chunk(label, split_id, sched->n_splits, chunk_id, split_backend, split, j0, j1);
                    }
                }

                return GGML_STATUS_SUCCESS;
            };

            if (profile_nodes) {
                int chunk_id = 0;
                int j0 = 0;
                for (int j1 = 0; j1 < split->graph.n_nodes; ++j1) {
                    const char * name = split->graph.nodes[j1]->name;
                    if (!ggml_sched_profile_node_boundary(name)) {
                        continue;
                    }

                    enum ggml_status ec = profile_compute_view(j0, j1 + 1, ++chunk_id, ggml_sched_profile_node_kind(name), true);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                    j0 = j1 + 1;
                }

                if (j0 < split->graph.n_nodes) {
                    enum ggml_status ec = profile_compute_view(j0, split->graph.n_nodes, ++chunk_id, "tail", true);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                }
            } else if (sched->moe_profile) {
                int chunk_id = 0;
                int j0 = 0;
                for (int j1 = 0; j1 < split->graph.n_nodes; ++j1) {
                    ggml_tensor * node = split->graph.nodes[j1];
                    if (node->op != GGML_OP_MUL_MAT_ID) {
                        continue;
                    }

                    enum ggml_status ec = profile_compute_view(j0, j1 + 1, ++chunk_id, "moe", false);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                    ggml_backend_sched_moe_profile_node(sched, split_backend, node, moe_profile_stats);
                    j0 = j1 + 1;
                }

                if (j0 < split->graph.n_nodes) {
                    enum ggml_status ec = profile_compute_view(j0, split->graph.n_nodes, ++chunk_id, "tail", false);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                }
            } else {
                int64_t t_start_us = 0;
                if (profile_splits) {
                    ggml_backend_synchronize(split_backend);
                    t_start_us = ggml_time_us();
                }

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                if (profile_splits) {
                    ggml_backend_synchronize(split_backend);

                    size_t input_bytes = 0;
                    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                        input_bytes += ggml_nbytes(split->inputs[input_id]);
                    }

                    const char * first = split->graph.n_nodes > 0 ? split->graph.nodes[0]->name : "";
                    const char * last  = split->graph.n_nodes > 0 ? split->graph.nodes[split->graph.n_nodes - 1]->name : "";

                    GGML_LOG(
                        "sched_profile: label=%s split=%d/%d backend=%s nodes=%d range=%d:%d inputs=%d input_bytes=%.2f MiB compute_ms=%.3f first=%s last=%s\n",
                        sched->profile_label[0] ? sched->profile_label : "-",
                        split_id + 1,
                        sched->n_splits,
                        ggml_backend_name(split_backend),
                        split->graph.n_nodes,
                        split->i_start,
                        split->i_end,
                        split->n_inputs,
                        input_bytes / 1048576.0,
                        (ggml_time_us() - t_start_us) / 1000.0,
                        first,
                        last);
                }
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    if (sched->moe_profile) {
        ggml_backend_sched_moe_profile_print(moe_profile_stats);
        ggml_backend_sched_moe_profile_dump_json(moe_profile_stats);
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    return ggml_backend_sched_compute_splits_range(sched, 0, -1);
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    const char * GGML_SCHED_PROFILE_COPY = getenv("GGML_SCHED_PROFILE_COPY");
    sched->profile_copy = GGML_SCHED_PROFILE_COPY ? atoi(GGML_SCHED_PROFILE_COPY) != 0 : false;
    const char * GGML_SCHED_PROFILE_COPY_DETAIL = getenv("GGML_SCHED_PROFILE_COPY_DETAIL");
    sched->profile_copy_detail = GGML_SCHED_PROFILE_COPY_DETAIL ? atoi(GGML_SCHED_PROFILE_COPY_DETAIL) != 0 : false;
    const char * GGML_SCHED_PROFILE_COPY_MAX = getenv("GGML_SCHED_PROFILE_COPY_MAX");
    sched->profile_copy_max = GGML_SCHED_PROFILE_COPY_MAX ? atoi(GGML_SCHED_PROFILE_COPY_MAX) : 0;
    sched->profile_copy_count = 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_reserve_node_range(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, int node_start, int node_end) {
    GGML_ASSERT(sched);
    GGML_ASSERT(measure_graph);
    GGML_ASSERT(node_start >= 0);
    GGML_ASSERT(node_end >= node_start);
    GGML_ASSERT(node_end <= measure_graph->n_nodes);

    struct ggml_cgraph measure_view = ggml_graph_view(measure_graph, node_start, node_end);
    return ggml_backend_sched_reserve(sched, &measure_view);
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

static int ggml_backend_sched_replace_tensor_refs(struct ggml_cgraph * graph, struct ggml_tensor * old_tensor, struct ggml_tensor * new_tensor) {
    int n_replaced = 0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node == old_tensor) {
            graph->nodes[i] = new_tensor;
            n_replaced++;
            continue;
        }
        if (node == nullptr) {
            continue;
        }

        if (node->view_src == old_tensor) {
            node->view_src = new_tensor;
            n_replaced++;
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (node->src[j] == old_tensor) {
                node->src[j] = new_tensor;
                n_replaced++;
            }
        }
    }

    for (int i = 0; i < graph->n_leafs; ++i) {
        if (graph->leafs[i] == old_tensor) {
            graph->leafs[i] = new_tensor;
            n_replaced++;
        }
    }

    return n_replaced;
}

static int ggml_backend_sched_next_nonzero_copy(ggml_backend_sched_t sched, int copy_id) {
    if (sched->n_copies <= 1) {
        return 0;
    }

    for (int i = 0; i < sched->n_copies; ++i) {
        const int candidate = (copy_id + i) % sched->n_copies;
        if (candidate != 0) {
            return candidate;
        }
    }

    return copy_id;
}

bool ggml_backend_sched_advance_copy(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    if (!sched->is_alloc || sched->n_copies <= 1 || sched->n_splits <= 0) {
        return false;
    }

    const int old_copy = sched->cur_copy;
    const int new_copy = ggml_backend_sched_next_nonzero_copy(sched, sched->next_copy);
    if (old_copy == new_copy) {
        return false;
    }

    if (old_copy == 0) {
        // Slot 0 aliases the original graph input tensors. Retire it once before
        // set_inputs() overwrites those tensors, then use duplicate slots only.
        for (int i = 0; i < sched->n_backends; ++i) {
            ggml_backend_synchronize(sched->backends[i]);
        }
    }

    int n_replaced = 0;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];
        const int backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            struct ggml_tensor * input = split->inputs[input_id];
            const size_t input_id_hash = hash_id(input);

            struct ggml_tensor * old_tensor = tensor_id_copy(input_id_hash, backend_id, old_copy);
            struct ggml_tensor * new_tensor = tensor_id_copy(input_id_hash, backend_id, new_copy);

            if (old_tensor == nullptr || new_tensor == nullptr || old_tensor == new_tensor) {
                continue;
            }

            n_replaced += ggml_backend_sched_replace_tensor_refs(&split->graph, old_tensor, new_tensor);
            n_replaced += ggml_backend_sched_replace_tensor_refs(&sched->graph, old_tensor, new_tensor);
        }
    }

    if (n_replaced == 0) {
        return false;
    }

    sched->cur_copy  = new_copy;
    sched->next_copy = ggml_backend_sched_next_nonzero_copy(sched, new_copy + 1);

    return true;
}

bool ggml_backend_sched_copy_graph_inputs(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    if (!sched->is_alloc || sched->n_copies <= 1) {
        return true;
    }

    for (int input_idx = 0; input_idx < sched->n_graph_inputs; ++input_idx) {
        struct ggml_tensor * input = sched->graph_inputs[input_idx];
        const size_t id = hash_id(input);
        const int backend_id = tensor_backend_id(input);
        if (backend_id < 0) {
            return false;
        }

        struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, sched->cur_copy);
        if (input_cpy == nullptr) {
            return false;
        }
        if (input_cpy == input) {
            continue;
        }

        ggml_backend_t backend = sched->backends[backend_id];
        if (sched->events[backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(backend);
        }

        ggml_backend_tensor_copy(input, input_cpy);
    }

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

enum ggml_status ggml_backend_sched_graph_compute_async_range(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int split_start, int split_end) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits_range(sched, split_start, split_end);
}

enum ggml_status ggml_backend_sched_graph_compute_async_node_range(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int node_start, int node_end) {
    GGML_ASSERT(sched);
    GGML_ASSERT(graph);
    GGML_ASSERT(node_start >= 0);
    GGML_ASSERT(node_end >= node_start);
    GGML_ASSERT(node_end <= graph->n_nodes);

    struct ggml_cgraph graph_view = ggml_graph_view(graph, node_start, node_end);

    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, &graph_view)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_profile(ggml_backend_sched_t sched, bool enabled, const char * label) {
    GGML_ASSERT(sched);

    sched->profile_splits = enabled;
    sched->profile_nodes = false;

    if (label && label[0]) {
        snprintf(sched->profile_label, sizeof(sched->profile_label), "%s", label);
    } else {
        sched->profile_label[0] = '\0';
    }
}

void ggml_backend_sched_set_profile_nodes(ggml_backend_sched_t sched, bool enabled, const char * label) {
    GGML_ASSERT(sched);

    sched->profile_splits = false;
    sched->profile_nodes = enabled;

    if (label && label[0]) {
        snprintf(sched->profile_label, sizeof(sched->profile_label), "%s", label);
    } else {
        sched->profile_label[0] = '\0';
    }
}

void ggml_backend_sched_set_moe_profile(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    sched->moe_profile = enabled;
}

void ggml_backend_sched_set_moe_profile_json(ggml_backend_sched_t sched, const char * path) {
    GGML_ASSERT(sched);
    sched->moe_profile_json = path ? path : "";
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_split_i_start(ggml_backend_sched_t sched, int split) {
    GGML_ASSERT(sched);
    GGML_ASSERT(split >= 0 && split < sched->n_splits);
    return sched->splits[split].i_start;
}

int ggml_backend_sched_get_split_i_end(ggml_backend_sched_t sched, int split) {
    GGML_ASSERT(sched);
    GGML_ASSERT(split >= 0 && split < sched->n_splits);
    return sched->splits[split].i_end;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
