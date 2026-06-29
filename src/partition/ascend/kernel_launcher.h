// src/partition/ascend/kernel_launcher.h
#ifndef DGL_ASCEND_KERNEL_LAUNCHER_H
#define DGL_ASCEND_KERNEL_LAUNCHER_H

#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "partition_kernel.h"

namespace dgl {
namespace partition {

class AscendCKernelLauncher {
public:
    explicit AscendCKernelLauncher(aclrtStream stream) : stream_(stream) {}

    // 启动取模分区核函数
    template<typename T>
    void LaunchMapProcByRemainder(const GlobalTensor<T>& global,
                                  GlobalTensor<T>& part_id,
                                  int64_t num_elements,
                                  int64_t num_parts) {
        MapProcByRemainderKernel<T> kernel;
        kernel.Init(global, part_id, num_elements, num_parts);

        // 配置执行参数
        int32_t block_dim = 256;
        int32_t block_num = (num_elements + block_dim - 1) / block_dim;

        // 启动核函数
        __aicore__ void (*kernel_func)() = [&]() { kernel.Process(); };
        aclrtLaunchKernel((void*)kernel_func, block_num, block_dim,
                          nullptr, 0, stream_);
    }

    // 启动本地映射核函数
    template<typename T>
    void LaunchMapLocalIndexByRemainder(const GlobalTensor<T>& global,
                                        GlobalTensor<T>& local,
                                        int64_t num_elements,
                                        int64_t num_parts) {
        MapLocalIndexByRemainderKernel<T> kernel;
        kernel.Init(global, local, num_elements, num_parts);

        int32_t block_dim = 256;
        int32_t block_num = (num_elements + block_dim - 1) / block_dim;

        __aicore__ void (*kernel_func)() = [&]() { kernel.Process(); };
        aclrtLaunchKernel((void*)kernel_func, block_num, block_dim,
                          nullptr, 0, stream_);
    }

    // 启动全局映射核函数
    template<typename T>
    void LaunchMapGlobalIndexByRemainder(const GlobalTensor<T>& local,
                                         GlobalTensor<T>& global,
                                         int64_t num_elements,
                                         int64_t num_parts,
                                         int64_t part_id) {
        MapGlobalIndexByRemainderKernel<T> kernel;
        kernel.Init(local, global, num_elements, num_parts, part_id);

        int32_t block_dim = 256;
        int32_t block_num = (num_elements + block_dim - 1) / block_dim;

        __aicore__ void (*kernel_func)() = [&]() { kernel.Process(); };
        aclrtLaunchKernel((void*)kernel_func, block_num, block_dim,
                          nullptr, 0, stream_);
    }

    // 启动直方图统计
    template<typename T>
    void LaunchComputeHistogram(const GlobalTensor<T>& data,
                                GlobalTensor<int64_t>& histogram,
                                int64_t num_elements,
                                int64_t num_bins) {
        ComputeHistogramKernel<T> kernel;
        kernel.Init(data, histogram, num_elements, num_bins);

        int32_t block_dim = 256;
        int32_t block_num = 256; // 最多256个block

        // 分配共享内存
        size_t shared_mem_size = num_bins * sizeof(int64_t);

        __aicore__ void (*kernel_func)() = [&]() { kernel.Process(); };
        aclrtLaunchKernel((void*)kernel_func, block_num, block_dim,
                          &shared_mem_size, 1, stream_);
    }

private:
    aclrtStream stream_;
};

} // namespace partition
} // namespace dgl

#endif // DGL_USE_ASCEND
#endif // DGL_ASCEND_KERNEL_LAUNCHER_H