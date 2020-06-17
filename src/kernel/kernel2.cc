/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/kernel2.cc
 * \brief New kernels
 */
#include "./kernel2.h"

#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>
#include <dgl/spfmt.h>

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
  if (GlobalSparseFormat::Get()->GetFormat() == SparseFormat::kCOO)
    format = SparseFormat::kCOO;
  else
    format = SparseFormat::kCSR;
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, {
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
           std::vector<NDArray> out_aux,
           SparseFormat format) {
  if (GlobalSparseFormat::Get()->GetFormat() == SparseFormat::kCSR)
    format = SparseFormat::kCSR;
  else
    format = SparseFormat::kCOO;
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(out->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, DType>(
              op, bcast, graph->GetCSRMatrix(0),
              ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, DType>(
              op, bcast, graph->GetCOOMatrix(0),
              ufeat, efeat, out, out_aux);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "E_data", "Out"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    SpMM(op, "sum", graph.sptr(), X, Y, Z, {});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUSum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {X, Z}, {"U_data", "Out"});
    SpMM("copy_u", "sum", graph.sptr(), X, aten::NullArray(), Z, {});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    NDArray argX = args[3];
    CheckCtx(graph->Context(), {X, Z, argX}, {"U_data", "Out", "U_index"});
    SpMM("copy_u", "max", graph.sptr(), X, aten::NullArray(),
         Z, {argX, aten::NullArray()});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    NDArray argX = args[3];
    CheckCtx(graph->Context(), {X, Z, argX}, {"U_data", "Out", "U_index"});
    SpMM("copy_u", "min", graph.sptr(), X, aten::NullArray(),
         Z, {argX, aten::NullArray()});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {Y, Z}, {"E_data", "Out"});
    SpMM("copy_e", "sum", graph.sptr(), aten::NullArray(), Y, Z, {
      aten::NullArray(), aten::NullArray()});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyEMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    NDArray argY = args[3];
    CheckCtx(graph->Context(), {Y, Z, argY}, {"E_data", "Out", "E_index"});
    SpMM("copy_e", "max", graph.sptr(), aten::NullArray(), Y,
         Z, {aten::NullArray(), argY});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyEMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    NDArray argY = args[3];
    CheckCtx(graph->Context(), {Y, Z, argY}, {"E_data", "Out", "E_index"});
    SpMM("copy_e", "min", graph.sptr(), aten::NullArray(), Y,
         Z, {aten::NullArray(), argY});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyU")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {X, Z}, {"U_data", "Out"});
    SDDMM("copy_u", graph.sptr(), X, aten::NullArray(), Z, {
      aten::NullArray(), aten::NullArray()});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpEMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
    SpMM(op, "max", graph.sptr(), X, Y, Z, {argX, argY});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpEMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
    SpMM(op, "min", graph.sptr(), X, Y, Z, {argX, argY});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpV")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "V_data", "Out"});
    SDDMM(op, graph.sptr(), X, Y, Z, {aten::NullArray(), aten::NullArray()});
  });

}  // namespace kernel
}  // namespace dgl
