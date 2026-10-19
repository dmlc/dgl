#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  IdType indptr_val = 0;
  auto device = runtime::DeviceAPI::Get(csr.indptr->ctx);
  device->CopyDataFromTo(
      static_cast<IdType*>(csr.indptr->data) + row, 0,
      &indptr_val, 0, sizeof(IdType),
      csr.indptr->ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  const int64_t offset = indptr_val * sizeof(IdType);
  return csr.indices.CreateView({len}, csr.indices->dtype, offset);
}

template NDArray CSRGetRowColumnIndices<kDGLAscend, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowColumnIndices<kDGLAscend, int64_t>(CSRMatrix, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  IdType indptr_val = 0;
  auto device = runtime::DeviceAPI::Get(csr.indptr->ctx);
  device->CopyDataFromTo(
      static_cast<IdType*>(csr.indptr->data) + row, 0,
      &indptr_val, 0, sizeof(IdType),
      csr.indptr->ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  const int64_t offset = indptr_val * sizeof(IdType);
  if (CSRHasData(csr))
    return csr.data.CreateView({len}, csr.data->dtype, offset);
  else
    return aten::Range(
        offset, offset + len, csr.indptr->dtype.bits, csr.indptr->ctx);
}

template NDArray CSRGetRowData<kDGLAscend, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowData<kDGLAscend, int64_t>(CSRMatrix, int64_t);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

