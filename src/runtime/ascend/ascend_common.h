/**
 *  Copyright (c) 2024 by Contributors
 * @file ascend_common.h
 * @brief Common utilities for Ascend NPU
 */
#ifndef DGL_RUNTIME_ASCEND_ASCEND_COMMON_H_
#define DGL_RUNTIME_ASCEND_ASCEND_COMMON_H_

#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <dgl/runtime/device_api.h>

#include <cstddef>

namespace dgl {
namespace runtime {

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error: " << aclGetRecentErrMsg(); \
  }

#define ASCEND_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)            \
  {                                                                           \
    if (!dgl::runtime::is_zero((nblks)) && !dgl::runtime::is_zero((nthrs))) { \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);          \
      aclError e = aclGetLastError();                                         \
      CHECK(e == ACL_SUCCESS) << "Ascend kernel launch error: " << e;         \
    }                                                                         \
  }

aclrtStream getCurrentAscendStream();
void setCurrentAscendStream(aclrtStream stream);

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_USE_ASCEND
#endif  // DGL_RUNTIME_ASCEND_ASCEND_COMMON_H_
