#ifndef DGL_KERNEL_BINARY_ELEWISE_H_
#define DGL_KERNEL_BINARY_ELEWISE_H_

#include <dgl/runtime/ndarray.h>

namespace dgl {
namespace kernel {
namespace binary_op {
  static const std::string kReduceSum = "sum";
  static const std::string kReduceMax = "max";
  static const std::string kReduceMin = "min";
  static const std::string kReduceMean = "mean";
  static const std::string kReduceProd = "prod";
  static const std::string kReduceNone = "none";
}  // namespace binary_op

/*
 * !\brief Infer the output shape of binary elewise graph computation.
 */
std::vector<int64_t> BinaryElewiseInferShape(
    const std::string& reducer,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray lhs,
    runtime::NDArray rhs);

/*
 * !\brief Multiply src node data with edge data and perform reduce
 * 
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param binary_op The type of the binary operator ("mul", "add").
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param edge_ids An optional int64 array for the edge ids. If empty,
 *                 the edge ids are consecutive integers [0, len(indices)).
 *                 The edge ids are used to read and write edge data.
 * \param src_data The source node feature tensor.
 * \param edge_data The edge feature tensor.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 */
runtime::NDArray SrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray edge_ids,
    runtime::NDArray src_data,
    runtime::NDArray edge_data);

/*
 * !\brief Multiply src node data with dst node data and perform reduce
 * 
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param binary_op The type of the mul functor ("mul", "add").
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param edge_ids An optional int64 array for the edge ids. If empty,
 *                 the edge ids are consecutive integers [0, len(indices)).
 *                 The edge ids are used to read and write edge data.
 * \param src_data The source node feature tensor.
 * \param dst_data The destination node feature tensor.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 */
runtime::NDArray SrcOpDstReduce(
    const std::string& reducer,
    const std::string& binary_op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray edge_ids,
    runtime::NDArray src_data,
    runtime::NDArray dst_data);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_ELEWISE_H_
