/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/shared_mem_manager.cc
 * \brief DGL sampler implementation
 */
#include "shared_mem_manager.h"

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/container.h>
#include <dgl/sampler.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "../c_api_common.h"
#include "heterograph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

// const size_t SHARED_MEM_METAINFO_SIZE_MAX = 1024 * 16;

template <>
NDArray SharedMemManager::CopyToSharedMem<NDArray>(const NDArray &data,
                                                   std::string name) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
  strm_->Write(data->ndim);
  strm_->Write(data->dtype);
  int ndim = data->ndim;
  strm_->WriteArray(data->shape, ndim);

  bool is_null = IsNullArray(data);
  strm_->Write(is_null);
  if (is_null) {
    return data;
  } else {
    return NDArray::EmptyShared(graph_name_ + name, shape, data->dtype, ctx,
                                true);
  }
}

template <>
CSRMatrix SharedMemManager::CopyToSharedMem<CSRMatrix>(const CSRMatrix &csr,
                                                       std::string name) {
  auto indptr_shared_mem = CopyToSharedMem(csr.indptr, name + "_indptr");
  auto indices_shared_mem = CopyToSharedMem(csr.indices, name + "_indices");
  auto data_shared_mem = CopyToSharedMem(csr.data, name + "_data");
  strm_->Write(csr.num_rows);
  strm_->Write(csr.num_cols);
  strm_->Write(csr.sorted);
  return CSRMatrix(csr.num_rows, csr.num_cols, indptr_shared_mem,
                   indices_shared_mem, data_shared_mem, csr.sorted);
}

template <>
COOMatrix SharedMemManager::CopyToSharedMem<COOMatrix>(const COOMatrix &coo,
                                                       std::string name) {
  auto row_shared_mem = CopyToSharedMem(coo.row, name + "_row");
  auto col_shared_mem = CopyToSharedMem(coo.col, name + "_col");
  auto data_shared_mem = CopyToSharedMem(coo.col, name + "_data");
  strm_->Write(coo.num_rows);
  strm_->Write(coo.num_cols);
  strm_->Write(coo.row_sorted);
  strm_->Write(coo.col_sorted);
  return COOMatrix(coo.num_rows, coo.num_cols, row_shared_mem, col_shared_mem,
                   data_shared_mem, coo.row_sorted, coo.col_sorted);
}

template <>
bool SharedMemManager::CreateFromSharedMem<NDArray>(NDArray *nd,
                                                    std::string name) {
  int ndim;
  DLContext ctx = {kDLCPU, 0};
  DLDataType dtype;

  CHECK(this->Read(&ndim)) << "Invalid DLTensor file format";
  CHECK(this->Read(&dtype)) << "Invalid DLTensor file format";

  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(this->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }
  *nd = NDArray::EmptyShared(graph_name_ + name, shape, dtype, ctx, false);
}

template <>
bool SharedMemManager::CreateFromSharedMem<COOMatrix>(COOMatrix *coo,
                                                      std::string name) {
  CreateFromSharedMem(&coo->row, name + "_row");
  CreateFromSharedMem(&coo->col, name + "_col");
  CreateFromSharedMem(&coo->data, name + "_data");
  strm_->Read(&coo->num_rows);
  strm_->Read(&coo->num_cols);
  strm_->Read(&coo->row_sorted);
  strm_->Read(&coo->col_sorted);
  return true;
}

template <>
bool SharedMemManager::CreateFromSharedMem<CSRMatrix>(CSRMatrix *csr,
                                                      std::string name) {
  CreateFromSharedMem(&csr->indices, name + "_indices");
  CreateFromSharedMem(&csr->indptr, name + "_indptr");
  CreateFromSharedMem(&csr->data, name + "_data");
  strm_->Read(&csr->num_rows);
  strm_->Read(&csr->num_cols);
  strm_->Read(&csr->sorted);
  return true;
}

// template <>
// HeteroGraphPtr SharedMemManager::CopyToSharedMem<HeteroGraphPtr>(
//   const HeteroGraphPtr &g, std::string name) {
//   auto hg = std::dynamic_pointer_cast<HeteroGraph>(g);
//   CHECK_NOTNULL(hg);
//   hg->SharedMemName();
//   if (hg->SharedMemName() == name) {
//     return g;
//   }

//   // Copy buffer to share memory
// //   auto mem = std::make_shared<SharedMemory>(name);
// //   auto mem_buf = mem->CreateNew(SHARED_MEM_METAINFO_SIZE_MAX);
// //   dmlc::MemoryFixedSizeStream ofs(mem_buf, SHARED_MEM_METAINFO_SIZE_MAX);
// //   dmlc::SeekStream *strm_ = &ofs;

//   bool has_coo = fmts.find("coo") != fmts.end();
//   bool has_csr = fmts.find("csr") != fmts.end();
//   bool has_csc = fmts.find("csc") != fmts.end();
//   strm_->Write(g->NumBits());
//   strm_->Write(has_coo);
//   strm_->Write(has_csr);
//   strm_->Write(has_csc);
//   strm_->Write(ImmutableGraph::ToImmutable(hg->meta_graph_));
//   strm_->Write(hg->num_verts_per_type_);

//   std::vector<HeteroGraphPtr> relgraphs(g->NumEdgeTypes());

//   for (dgl_type_t etype = 0 ; etype < g->NumEdgeTypes() ; ++etype) {
//     strm_->Write(hg->NumEdges(etype));
//     aten::COOMatrix coo;
//     aten::CSRMatrix csr, csc;
//     std::string prefix = name + "_" + std::to_string(etype);
//     if (has_coo) {
//       coo = hg->GetCOOMatrix(etype).CopyToSharedMem(prefix + "_coo");
//       strm_->Write(coo.row_sorted);
//       strm_->Write(coo.col_sorted);
//     }
//     if (has_csr) {
//       csr = hg->GetCSRMatrix(etype).CopyToSharedMem(prefix + "_csr");
//       strm_->Write(csr.sorted);
//     }
//     if (has_csc) {
//       csc = hg->GetCSCMatrix(etype).CopyToSharedMem(prefix + "_csc");
//       strm_->Write(csc.sorted);
//     }
//     relgraphs[etype] = UnitGraph::CreateHomographFrom(csc, csr, coo, has_csc,
//     has_csr, has_coo);
//   }

//   auto ret = std::shared_ptr<HeteroGraph>(
//       new HeteroGraph(hg->meta_graph_, relgraphs, hg->num_verts_per_type_));
//   ret->shared_mem_ = mem;

//   strm_->Write(ntypes);
//   strm_->Write(etypes);
//   return ret;
// }

}  // namespace dgl
