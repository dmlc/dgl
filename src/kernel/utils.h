#ifndef DGL_KERNEL_UTILS_H_
#define DGL_KERNEL_UTILS_H_

#include <cstdlib>
#include <dlpack/dlpack.h>
#include <dgl/runtime/ndarray.h>

namespace minigun {
struct Csr;
}  // namespace minigun

namespace dgl {
namespace kernel {
namespace utils {

/* !\brief Return an NDArray that represents none value. */
inline runtime::NDArray NoneArray() {
  return runtime::NDArray::Empty({}, DLDataType{kDLInt, 32, 1}, DLContext{kDLCPU, 0});
}

/* !\brief Return true if the NDArray is none. */
inline bool IsNoneArray(runtime::NDArray array) {
  return array->ndim == 0;
}

/*
 * !\brief Find number of threads is smaller than dim and max_nthrs
 * and is also the power of two.
 */
int FindNumThreads(int dim, int max_nthrs);

/*
 * !\brief Compute the total number of feature elements.
 */
int64_t ComputeXLength(runtime::NDArray feat_array);

/*
 * !\brief Compute the total number of elements in the array.
 */
int64_t NElements(const runtime::NDArray& array);

/*
 * !\brief Compute the product of the given vector.
 */
int64_t Prod(const std::vector<int64_t>& vec);

/*
 * !\brief Create minigun CSR from two ndarrays.
 */
minigun::Csr CreateCsr(runtime::NDArray indptr, runtime::NDArray indices);

/*
 * !\brief Fill the array with constant value.
 */
template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val);

}  // namespace utils
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_UTILS_H_
