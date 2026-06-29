// src/partition/ascend/partition_kernel.cpp
#include "partition_kernel.h"

using namespace AscendC;

namespace dgl {
namespace partition {
namespace kernels {

// 模板实例化 - Remainder kernels
template class MapProcByRemainderKernel<int32_t>;
template class MapProcByRemainderKernel<int64_t>;

template class MapLocalIndexByRemainderKernel<int32_t>;
template class MapLocalIndexByRemainderKernel<int64_t>;

template class MapGlobalIndexByRemainderKernel<int32_t>;
template class MapGlobalIndexByRemainderKernel<int64_t>;

// 直方图实例化
template class ComputeHistogramKernel<int32_t>;
template class ComputeHistogramKernel<int64_t>;

} // namespace kernels
} // namespace partition
} // namespace dgl