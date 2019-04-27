/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/kernel_apis.cc
 * \brief kernel APIs for graph computation
 */
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

/*
 * !\brief Copy edge data and perform reduce
 * 
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param indptr An int64 row offset array for the graph CSR.
 * \param indices An int64 column index array for the graph CSR.
 * \param edge_ids An optional int64 array for the edge ids. If empty,
 *                 the edge ids are consecutive integers [0, len(indices)).
 *                 The edge ids are used to read and write edge data.
 * \param edge_data The source node feature tensor.
 * \return out_data The output tensor. Could be either node or edge feature tensor
 *                  depending on the reducer.
 */
DGL_REGISTER_GLOBAL("backend.kernel._CAPI_DGLKernelCopyEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    LOG(FATAL) << "Not implemented";
  });

}  // namespace dgl
