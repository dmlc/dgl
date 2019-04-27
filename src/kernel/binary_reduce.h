#ifndef DGL_KERNEL_BINARY_REDUCE_H_
#define DGL_KERNEL_BINARY_REDUCE_H_

#include <dgl/runtime/ndarray.h>
#include "./binary_reduce_common.h"

namespace dgl {
namespace kernel {

// Structure for broadcasting shapes
struct BcastInfo {
  // inferred output shape
  std::vector<int64_t> real_out_shape;
  // Following shapes here have been preprocessed, so that:
  //  - The first dimension (for graph) is removed. Shapes here are only for features.
  //  - They have the same number of dimensions.
  //    e.g. (4,) and (3, 4) become (1, 4) and (3, 4)
  //  - Continuous non-broadcasting dimenions are flattened.
  //    e.g. (4, 1, 3, 3) and (4, 5, 3, 3) become (4, 1, 9) and (4, 5, 9)
  std::vector<int64_t> lhs_shape, lhs_stride;
  std::vector<int64_t> rhs_shape, rhs_stride;
  std::vector<int64_t> out_shape, out_stride;
};

/*
 * !\brief Multiply src node data with edge data and perform reduce
 *
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param binary_op The type of the binary operator ("mul", "add").
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param src_mapping An optional int64 array for source node mapping. If empty,
 *                    source ids are consecutive integers [0, len(indptr) - 1).
 *                    Source ids are used to read source node data.
 * \param edge_mapping An optional int64 array for edge mapping. If empty,
 *                     the edge ids are consecutive integers [0, len(indices)).
 *                     The edge ids are used to read edge data.
 * \param src_data The source node feature tensor.
 * \param edge_data The edge feature tensor.
 * \param out_mapping An optional int64 array for output mapping. If reducer is
 *                    "none", then it's a mapping to edge ids. Otherwise, it's
 *                    mapping to destination node ids.
 * \param out_size An integer indicating the output size. If reducer is "none",
 *                 it is the number of output edges. Otherwise it's the number
 *                 of output nodes.
 * \return out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 */
runtime::NDArray SrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray src_mapping,
    runtime::NDArray edge_mapping,
    runtime::NDArray src_data,
    runtime::NDArray edge_data,
    runtime::NDArray out_mapping,
    const int64_t out_size);

/*
 * !\brief Multiply src node data with dst node data and perform reduce
 *
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param binary_op The type of the mul functor ("mul", "add").
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param src_mapping An optional int64 array for source node mapping. If empty,
 *                    source ids are consecutive integers [0, len(indptr) - 1).
 *                    Source ids are used to read source node data.
 * \param dst_mapping An optional int64 array for destination node mapping.
 *                    If empty, the destination ids are consecutive integers
 *                    [0, len(indptr) - 1). The destination ids are used to
 *                    read destination node data.
 * \param src_data The source node feature tensor.
 * \param dst_data The destination node feature tensor.
 * \param out_mapping An optional int64 array for output mapping. If reducer is
 *                    "none", then it's a mapping to edge ids. Otherwise, it's
 *                    mapping to destination node ids.
 * \param out_size An integer indicating the output size. If reducer is "none",
 *                 it is the number of output edges. Otherwise it's the number
 *                 of output nodes.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 */
runtime::NDArray SrcOpDstReduce(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray src_mapping,
    runtime::NDArray dst_mapping,
    runtime::NDArray src_data,
    runtime::NDArray dst_data,
    runtime::NDArray out_mapping,
    const int64_t out_size);


/*
 * !\brief Copy src node data and perform reduce
 
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param src_mapping An optional int64 array for source node mapping. If empty,
 *                    source ids are consecutive integers [0, len(indptr) - 1).
 *                    Source ids are used to read source node data.
 * \param src_data The source node feature tensor.
 * \param out_mapping An optional int64 array for output mapping. If reducer is
 *                    "none", then it's a mapping to edge ids. Otherwise, it's
 *                    mapping to destination node ids.
 * \param out_size An integer indicating the output size. If reducer is "none",
 *                 it is the number of output edges. Otherwise it's the number
 *                 of output nodes.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 *
 */
runtime::NDArray CopySrcReduce(
    const std::string& reducer,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray src_mapping,
    runtime::NDArray src_data,
    runtime::NDArray out_mapping,
    const int64_t out_size);

/*
 * !\brief Copy edge data and perform reduce
 
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param edge_mapping An optional int64 array for source node mapping. If empty,
 *                    source ids are consecutive integers [0, len(indptr) - 1).
 *                    Source ids are used to read source node data.
 * \param edge_data The source node feature tensor.
 * \param out_mapping An optional int64 array for output mapping. If reducer is
 *                    "none", then it's a mapping to edge ids. Otherwise, it's
 *                    mapping to destination node ids.
 * \param out_size An integer indicating the output size. If reducer is "none",
 *                 it is the number of output edges. Otherwise it's the number
 *                 of output nodes.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 *
 */
runtime::NDArray CopyEdgeReduce(
    const std::string& reducer,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray edge_mapping,
    runtime::NDArray edge_data,
    runtime::NDArray out_mapping,
    const int64_t out_size);


// Declaration of implementations.

namespace cpu {
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data);


void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data);
}  // namespace cpu

namespace cuda {
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data);

void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data);
}  // namespace cuda

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_H_
