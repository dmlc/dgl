/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/kernel2.cc
 * \brief New kernels
 */
#include "./kernel2.h"

#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace kernel {
namespace {

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (aten::IsNullArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

}  // namespace

void SpMM(const std::string& op, const std::string& reduce,
          HeteroGraphPtr graph,
          NDArray ufeat,
          NDArray efeat,
          NDArray out,
          std::vector<NDArray> out_aux,
          SparseFormat format) {
  // TODO(zihao): format tuning
  format = SparseFormat::kCSR;
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(out->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SpMMCsr<XPU, IdType, DType>(
              op, reduce, bcast, graph->GetCSCMatrix(0),
              ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SpMMCoo<XPU, IdType, DType>(
              op, reduce, bcast, graph->GetCOOMatrix(0),
              ufeat, efeat, out, out_aux);
        } else {
          LOG(FATAL) << "SpMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

void SDDMM(const std::string& op,
           HeteroGraphPtr graph,
           NDArray ufeat,
           NDArray efeat,
           NDArray out,
           SparseFormat format) {
  // TODO(zihao): format tuning
  format = SparseFormat::kCOO;
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(out->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, DType>(
              op, bcast, graph->GetCSRMatrix(0),
              ufeat, efeat, out);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, DType>(
              op, bcast, graph->GetCOOMatrix(0),
              ufeat, efeat, out);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelSpMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    const std::string reduce_op = args[2];
    NDArray U = args[3];
    NDArray E = args[4];
    NDArray V = args[5];
    NDArray ArgU = args[6];
    NDArray ArgE = args[7];
    CheckCtx(graph->Context(), {U, E, V, ArgU, ArgE},
        {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    SpMM(op, reduce_op, graph.sptr(), U, E, V, {ArgU, ArgE});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelSDDMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    NDArray U = args[2];
    NDArray V = args[3];
    NDArray E = args[4];
    CheckCtx(graph->Context(), {U, V, E}, {"U_data", "V_data", "out"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    SDDMM(op, graph.sptr(), U, V, E);
  });

}  // namespace kernel
}  // namespace dgl
