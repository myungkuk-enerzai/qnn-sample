#include "QnnSharedBuffer.h"
#include <dlfcn.h>

constexpr uint8_t RPCMEM_HEAP_ID_SYSTEM = 25;
constexpr uint8_t RPCMEM_DEFAULT_FLAGS = 1;

std::mutex SharedBuffer::init_mutex_;

intptr_t align_to(size_t alignment, intptr_t offset)
{
    return offset % alignment == 0 ? offset
                                   : offset +
                                         (static_cast<intptr_t>(alignment) -
                                          offset % static_cast<intptr_t>(alignment));
}

SharedBuffer::~SharedBuffer()
{
    unload();
}

SharedBuffer &SharedBuffer::get_shared_buffer_manager()
{
    std::lock_guard<std::mutex> lock(init_mutex_);
    static SharedBuffer manager;
    if (!manager.get_init())
    {
        manager.load();
        manager.set_init(true);
    }

    return manager;
}

void *SharedBuffer::get_custom_memory_base(void *buf)
{
    auto iter = tensor_addr_to_custom_mem.find(buf);
    if (iter == tensor_addr_to_custom_mem.end())
    {
        return nullptr;
    }

    return iter->second;
}

void *SharedBuffer::get_unaligned_addr(void *buf)
{
    auto iter = restore_map_.find(buf);
    if (iter == restore_map_.end())
    {
        return nullptr;
    }

    return iter->second;
}

void *SharedBuffer::allocmem(uint32_t bytes, uint32_t align)
{
    if (!initialized_)
        return nullptr;

    uint32_t alloc_bytes = (bytes + align);
    void *buf = rpc_mem_alloc_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, alloc_bytes);
    if (!buf)
        return nullptr;

    alloc_size_map_.insert({buf, alloc_bytes});

    void *aligned_buf = reinterpret_cast<void *>(align_to(align, reinterpret_cast<intptr_t>(buf)));
    bool status = restore_map_.insert({aligned_buf, buf}).second;
    if (!status)
    {
        rpc_mem_free_(buf);
        return nullptr;
    }

    return aligned_buf;
}

void SharedBuffer::freemem(void *buf)
{
    if (!initialized_)
        return;
    else if (restore_map_.count(buf) == 0)
        printf("Don't free an unallocated tensor.");
    else
    {
        rpc_mem_free_(restore_map_[buf]);
        restore_map_.erase(buf);
    }
}

int32_t SharedBuffer::mem2fd(void *buf)
{
    int32_t memfd = -1;
    if (!initialized_)
        return -1;
    memfd = rpc_mem_to_fd_(buf);
    return memfd;
}

bool SharedBuffer::is_allocated(void *buf)
{
    return restore_map_.count(buf) != 0U;
}

void SharedBuffer::add_custom_memory_tensor_addr(void *tensor_addr, void *custom_mem)
{
    tensor_addr_to_custom_mem.insert({tensor_addr, custom_mem});
}

void SharedBuffer::load()
{
    lib_cdsp_rpc_ = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_GLOBAL);

    printf("lib_cdsp_rpc_: %p\n", lib_cdsp_rpc_);

    if (!lib_cdsp_rpc_)
    {
        printf("Failed to load libcdsprpc.so %s\n", dlerror());
        return;
    }

    rpc_mem_alloc_ = reinterpret_cast<RpcMemAllocFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_alloc"));
    rpc_mem_free_ = reinterpret_cast<RpcMemFreeFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_free"));
    rpc_mem_to_fd_ = reinterpret_cast<RpcMemToFdFn_t>(dlsym(lib_cdsp_rpc_, "rpcmem_to_fd"));
    if (nullptr == rpc_mem_alloc_ || nullptr == rpc_mem_free_ || nullptr == rpc_mem_to_fd_)
    {
        printf("Unable to access symbols in shared buffer. dlerror(): %s", dlerror());
        dlclose(lib_cdsp_rpc_);
        return;
    }
}

void SharedBuffer::unload()
{
    if (dlclose(lib_cdsp_rpc_) != 0)
    {
        printf("Failed to close libcdsprpc.so %s\n", dlerror());
    }
}
