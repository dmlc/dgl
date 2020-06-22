/*!
 *  Copyright (c) 2019 by Contributors
 * \file geometry/geometry_op.h
 * \brief Geometry operator templates
 */
#ifndef DGL_GEOMETRY_GEOMETRY_OP_H_
#define DGL_GEOMETRY_GEOMETRY_OP_H_

#include <dgl/array.h>

namespace dgl {
namespace geometry {
namespace impl {

template <DLDeviceType XPU, typename FloatType, typename IdType>
void FarthestPointSampler(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl

#endif  // DGL_GEOMETRY_GEOMETRY_OP_H_
