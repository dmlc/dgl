#ifndef DGL_KERNEL_BINARY_REDUCE_H_
#define DGL_KERNEL_BINARY_REDUCE_H_

#include <dgl/runtime/ndarray.h>
#include "./common.h"

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

template <int XPU, typename DType, typename EidGetter,
          typename OutSelector, typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BinaryReduceExecutor {
  static void Run(
    runtime::NDArray indptr,
    runtime::NDArray indices,
    runtime::NDArray edge_ids,
    runtime::NDArray src_data,
    runtime::NDArray edge_data,
    runtime::NDArray dst_data,
    runtime::NDArray out_data);
};

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

// common binary functors
template <typename DType>
struct BinaryAdd {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs + rhs;
  }
};

template <typename DType>
struct BinaryMul {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs * rhs;
  }
};

template <typename DType>
struct BinarySub {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs - rhs;
  }
};

template <typename DType>
struct BinaryDiv {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs / rhs;
  }
};

template <typename DType>
struct BinaryUseLhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs;
  }
};

template <typename DType>
struct BinaryUseRhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return rhs;
  }
};

#define BINARY_OP_SWITCH(val, DType, OpType, ...)   \
  if (val == "add") {                               \
    typedef BinaryAdd<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "sub") {                        \
    typedef BinarySub<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "mul") {                        \
    typedef BinaryMul<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "div") {                        \
    typedef BinaryDiv<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else {                                          \
    LOG(FATAL) << "Unsupported binary op: " << val; \
  }

// functors for reducers
template <int XPU, typename DType>
struct ReduceSum { };

template <int XPU, typename DType>
struct ReduceMax { };

template <int XPU, typename DType>
struct ReduceMin { };

template <int XPU, typename DType>
struct ReduceMean { };

template <int XPU, typename DType>
struct ReduceProd { };

template <int XPU, typename DType>
struct ReduceNone { };

#define REDUCER_SWITCH(val, XPU, DType, RedType, ...)   \
  if (val == binary_op::kReduceSum) {              \
    typedef ReduceSum<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMax) {       \
    typedef ReduceMax<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMin) {       \
    typedef ReduceMin<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMean) {      \
    typedef ReduceMean<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceProd) {      \
    typedef ReduceProd<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceNone) {      \
    typedef ReduceNone<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "Unsupported reducer: " << val;  \
  }

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_H_
