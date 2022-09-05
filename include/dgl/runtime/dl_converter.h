/*!
 *  Copyright (c) 2022 by Contributors
 * \file include/dgl/runtime/dl_converter.h
 * \brief DLPack converter.
 */
#ifndef DGL_RUNTIME_DL_CONVERTER_H_
#define DGL_RUNTIME_DL_CONVERTER_H_

#include "c_runtime_api.h"
#include "ndarray.h"
#include <dlpack/dlpack.h>

namespace dgl {
namespace runtime {

struct DLConverter {
  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * \param tensor The DLPack tensor to copy from.
   * \return The created NDArray view.
   */
  DGL_DLL static NDArray FromDLPack(DLManagedTensor* tensor);

  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of DGL.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(NDArray::Container* ptr);

  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(NDArray::Container* from);

  // NDArray to DLManagedTensor
  static DLManagedTensor* ToDLPack(const NDArray &from);
};

}  // namespace runtime
}  // namespace dgl

#ifdef __cplusplus
extern "C" {
#endif
/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayFromDLPack(DLManagedTensor* from,
                               DGLArrayHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayToDLPack(DGLArrayHandle from, DLManagedTensor** out,
                             int alignment = 0);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
DGL_DLL void DGLDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

#ifdef __cplusplus
}  // DGL_EXTERN_C
#endif
#endif  // DGL_RUNTIME_DL_CONVERTER_H_
