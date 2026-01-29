#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <unordered_set>

using RpcMemAllocFn_t = void *(*)(int, uint32_t, int);
using RpcMemFreeFn_t = void (*)(void *);
using RpcMemToFdFn_t = int (*)(void *);

class SharedBuffer
{
private:
    SharedBuffer() = default;

public:
    SharedBuffer(const SharedBuffer &) = delete;
    SharedBuffer &operator=(const SharedBuffer &) = delete;
    SharedBuffer(SharedBuffer &&) = delete;
    SharedBuffer &operator=(SharedBuffer &&) = delete;
    ~SharedBuffer();

    static SharedBuffer &get_shared_buffer_manager();

    bool get_init() { return initialized_; }
    void set_init(bool init) { initialized_ = init; }

    void *allocmem(uint32_t bytes, uint32_t align);
    void freemem(void *buf);
    int32_t mem2fd(void* buf);

    bool is_allocated(void *buf);

private:
    void load();
    void unload();

private:
    static std::mutex init_mutex_;

    bool initialized_{false};
    void *lib_cdsp_rpc_{nullptr};

    // Function pointer to rpcmem_alloc
    RpcMemAllocFn_t rpc_mem_alloc_;

    // Function pointer to rpcmem_free
    RpcMemFreeFn_t rpc_mem_free_;

    // Function pointer to rpcmem_to_fd
    RpcMemToFdFn_t rpc_mem_to_fd_;

    // Maps for the custom memory
    std::unordered_map<void *, void *> restore_map_;
    std::unordered_map<void *, uint32_t> alloc_size_map_;
};