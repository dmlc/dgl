/**
 *  Copyright (c) 2022 by Contributors
 * @file include/dgl/runtime/dlpack_convert.h
 * @brief Conversion between NDArray and DLPack.
 */
#ifndef DGL_RUNTIME_DLPACK_CONVERT_H_
#define DGL_RUNTIME_DLPACK_CONVERT_H_

#include "c_runtime_api.h"
#include "ndarray.h"

struct DLManagedTensor;

namespace dgl {
namespace runtime {

struct DLPackConvert {
  /**
   * @brief Create a DGL NDArray from a DLPack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * @param tensor The DLPack tensor to copy from.
   * @return The created NDArray view.
   */
  static NDArray FromDLPack(DLManagedTensor* tensor);

  /**
   * @brief Deleter for NDArray converted from DLPack.
   *
   * This is used from data which is passed from external
   * DLPack(DLManagedTensor) that are not allocated inside of DGL. This enables
   * us to create NDArray from memory allocated by other frameworks that are
   * DLPack compatible
   */
  static void DLPackDeleter(NDArray::Container* ptr);

  /** @brief Convert a DGL NDArray to a DLPack tensor.
   *
   * @param from The DGL NDArray.
   * @return A DLPack tensor.
   */
  static DLManagedTensor* ToDLPack(const NDArray& from);
};

}  // namespace runtime
}  // namespace dgl

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Delete (free) a DLManagedTensor's data.
 * @param dltensor Pointer to the DLManagedTensor.
 */
DGL_DLL void DGLDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/**
 * @brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * @param from The source DLManagedTensor.
 * @param out The output array handle.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayFromDLPack(DLManagedTensor* from, DGLArrayHandle* out);

/**
 * @brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * @param from The source array.
 * @param out The DLManagedTensor handle.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayToDLPack(
    DGLArrayHandle from, DLManagedTensor** out, int alignment = 0);

#ifdef __cplusplus
}  // DGL_EXTERN_C
#endif
#endif  // DGL_RUNTIME_DLPACK_CONVERT_H_
