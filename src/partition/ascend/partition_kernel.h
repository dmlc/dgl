// src/partition/ascend/partition_kernel.h
#ifndef DGL_ASCEND_PARTITION_KERNEL_H
#define DGL_ASCEND_PARTITION_KERNEL_H

#include "kernel_operator.h"

// 使用 AscendC 命名空间
using namespace AscendC;

namespace dgl {
namespace partition {
namespace kernels {

// 取模分区核函数
template<typename T>
class MapProcByRemainderKernel {
public:
    __aicore__ MapProcByRemainderKernel() {}

    __aicore__ void Init(const GlobalTensor<T>& global,
                         GlobalTensor<T>& part_id,
                         int64_t num_elements,
                         int64_t num_parts) {
        this->global_ = global;
        this->part_id_ = part_id;
        this->num_elements_ = num_elements;
        this->num_parts_ = num_parts;
    }

    __aicore__ void Process() {
        int32_t block_idx = GetBlockIdx();
        int32_t block_dim = GetBlockNum();
        int32_t thread_idx = GetThreadIdx();

        int64_t tid = block_idx * block_dim + thread_idx;
        if (tid < num_elements_) {
            T val = global_.GetValue(tid);
            part_id_.SetValue(tid, val % num_parts_);
        }
    }

private:
    GlobalTensor<T> global_;
    GlobalTensor<T> part_id_;
    int64_t num_elements_;
    int64_t num_parts_;
};

// 本地索引映射核函数
template<typename T>
class MapLocalIndexByRemainderKernel {
public:
    __aicore__ MapLocalIndexByRemainderKernel() {}

    __aicore__ void Init(const GlobalTensor<T>& global,
                         GlobalTensor<T>& local,
                         int64_t num_elements,
                         int64_t num_parts) {
        this->global_ = global;
        this->local_ = local;
        this->num_elements_ = num_elements;
        this->num_parts_ = num_parts;
    }

    __aicore__ void Process() {
        int32_t block_idx = GetBlockIdx();
        int32_t block_dim = GetBlockNum();
        int32_t thread_idx = GetThreadIdx();

        int64_t tid = block_idx * block_dim + thread_idx;
        if (tid < num_elements_) {
            T val = global_.GetValue(tid);
            local_.SetValue(tid, val / num_parts_);
        }
    }

private:
    GlobalTensor<T> global_;
    GlobalTensor<T> local_;
    int64_t num_elements_;
    int64_t num_parts_;
};

// 全局索引映射核函数
template<typename T>
class MapGlobalIndexByRemainderKernel {
public:
    __aicore__ MapGlobalIndexByRemainderKernel() {}

    __aicore__ void Init(const GlobalTensor<T>& local,
                         GlobalTensor<T>& global,
                         int64_t num_elements,
                         int64_t num_parts,
                         int64_t part_id) {
        this->local_ = local;
        this->global_ = global;
        this->num_elements_ = num_elements;
        this->num_parts_ = num_parts;
        this->part_id_ = part_id;
    }

    __aicore__ void Process() {
        int32_t block_idx = GetBlockIdx();
        int32_t block_dim = GetBlockNum();
        int32_t thread_idx = GetThreadIdx();

        int64_t tid = block_idx * block_dim + thread_idx;
        if (tid < num_elements_) {
            T val = local_.GetValue(tid);
            global_.SetValue(tid, val * num_parts_ + part_id_);
        }
    }

private:
    GlobalTensor<T> local_;
    GlobalTensor<T> global_;
    int64_t num_elements_;
    int64_t num_parts_;
    int64_t part_id_;
};

// 直方图统计核函数
template<typename T>
class ComputeHistogramKernel {
public:
    __aicore__ ComputeHistogramKernel() {}

    __aicore__ void Init(const GlobalTensor<T>& data,
                         GlobalTensor<int64_t>& histogram,
                         int64_t num_elements,
                         int64_t num_bins) {
        this->data_ = data;
        this->histogram_ = histogram;
        this->num_elements_ = num_elements;
        this->num_bins_ = num_bins;
    }

    __aicore__ void Process() {
        // 使用 UB 作为本地直方图
        __shared__ int64_t local_hist[1024];

        int32_t tid = GetThreadIdx();
        int32_t bid = GetBlockIdx();
        int32_t block_dim = GetBlockNum();

        // 初始化本地直方图
        for (int i = tid; i < num_bins_; i += block_dim) {
            local_hist[i] = 0;
        }
        __syncthreads();

        // 计算当前 block 处理的范围
        int64_t block_size = (num_elements_ + GetBlockNum() - 1) / GetBlockNum();
        int64_t start = bid * block_size;
        int64_t end = min(start + block_size, num_elements_);

        // 统计
        for (int64_t i = start + tid; i < end; i += block_dim) {
            T val = data_.GetValue(i);
            if (val >= 0 && val < num_bins_) {
                atomicAdd(&local_hist[val], 1);
            }
        }
        __syncthreads();

        // 合并到全局直方图
        for (int i = tid; i < num_bins_; i += block_dim) {
            if (local_hist[i] > 0) {
                atomicAdd(&histogram_.GetValue(i), local_hist[i]);
            }
        }
    }

private:
    GlobalTensor<T> data_;
    GlobalTensor<int64_t> histogram_;
    int64_t num_elements_;
    int64_t num_bins_;
};

} // namespace kernels
} // namespace partition
} // namespace dgl

#endif // DGL_ASCEND_PARTITION_KERNEL_H