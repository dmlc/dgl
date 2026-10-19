// ============================================================================
// CSRTranspose Ascend 实现 — CSR 转置（等价于 CSR→CSC）
// ============================================================================
//
// 实现策略：复用已有 Ascend 算子链路
//   CSRTranspose(csr) = COOToCSR(COOTranspose(CSRToCOO(csr, false)))
//
// 依赖的 Ascend 已适配算子：
//   - CSRToCOO<kDGLAscend, IdType>  (src/array/ascend/csr_to_coo.cc)
//   - COOTranspose                   (纯元数据交换，无数据拷贝)
//   - COOToCSR<kDGLAscend, IdType>  (src/array/ascend/coo2csr.cc)
//
// 与 CUDA int64 路径一致（cuda/csr_transpose.cc:86-88）。
// ============================================================================

#include <dgl/array.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <>
CSRMatrix CSRTranspose<kDGLAscend, int32_t>(CSRMatrix csr) {
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
}

template <>
CSRMatrix CSRTranspose<kDGLAscend, int64_t>(CSRMatrix csr) {
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl
