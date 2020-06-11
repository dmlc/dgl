#ifndef DGL_PCL_FPS_CUDA_H_
#define DGL_PCL_FPS_CUDA_H_

#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"

namespace dgl {
namespace aten {
namespace cuda {

template <typename DType>
IdArray _FPS_CUDA(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);

} // cuda
} // atem
} // dgl

#endif // DGL_PCL_FPS_CUDA_H_
