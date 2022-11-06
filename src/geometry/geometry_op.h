/**
 *  Copyright (c) 2019 by Contributors
 * @file geometry/geometry_op.h
 * @brief Geometry operator templates
 */
#ifndef DGL_GEOMETRY_GEOMETRY_OP_H_
#define DGL_GEOMETRY_GEOMETRY_OP_H_

#include <dgl/array.h>

namespace dgl {
namespace geometry {
namespace impl {

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void FarthestPointSampler(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result);

/** @brief Implementation of weighted neighbor matching process of edge
 * coarsening used in Metis and Graclus for homogeneous graph coarsening. This
 * procedure keeps picking an unmarked vertex and matching it with one its
 * unmarked neighbors (that maximizes its edge weight) until no match can be
 * done.
 */
template <DGLDeviceType XPU, typename FloatType, typename IdType>
void WeightedNeighborMatching(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);

/** @brief Implementation of neighbor matching process of edge coarsening used
 * in Metis and Graclus for homogeneous graph coarsening. This procedure keeps
 * picking an unmarked vertex and matching it with one its unmarked neighbors
 * (that maximizes its edge weight) until no match can be done.
 */
template <DGLDeviceType XPU, typename IdType>
void NeighborMatching(const aten::CSRMatrix &csr, IdArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl

#endif  // DGL_GEOMETRY_GEOMETRY_OP_H_
