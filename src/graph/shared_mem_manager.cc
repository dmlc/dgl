/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/shared_mem_manager.cc
 * @brief DGL sampler implementation
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

template <>
NDArray SharedMemManager::CopyToSharedMem<NDArray>(
    const NDArray &data, std::string name) {
  DGLContext ctx = {kDGLCPU, 0};
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
    auto nd =
        NDArray::EmptyShared(graph_name_ + name, shape, data->dtype, ctx, true);
    nd.CopyFrom(data);
    return nd;
  }
}

template <>
CSRMatrix SharedMemManager::CopyToSharedMem<CSRMatrix>(
    const CSRMatrix &csr, std::string name) {
  auto indptr_shared_mem = CopyToSharedMem(csr.indptr, name + "_indptr");
  auto indices_shared_mem = CopyToSharedMem(csr.indices, name + "_indices");
  auto data_shared_mem = CopyToSharedMem(csr.data, name + "_data");
  strm_->Write(csr.num_rows);
  strm_->Write(csr.num_cols);
  strm_->Write(csr.sorted);
  return CSRMatrix(
      csr.num_rows, csr.num_cols, indptr_shared_mem, indices_shared_mem,
      data_shared_mem, csr.sorted);
}

template <>
COOMatrix SharedMemManager::CopyToSharedMem<COOMatrix>(
    const COOMatrix &coo, std::string name) {
  auto row_shared_mem = CopyToSharedMem(coo.row, name + "_row");
  auto col_shared_mem = CopyToSharedMem(coo.col, name + "_col");
  auto data_shared_mem = CopyToSharedMem(coo.data, name + "_data");
  strm_->Write(coo.num_rows);
  strm_->Write(coo.num_cols);
  strm_->Write(coo.row_sorted);
  strm_->Write(coo.col_sorted);
  return COOMatrix(
      coo.num_rows, coo.num_cols, row_shared_mem, col_shared_mem,
      data_shared_mem, coo.row_sorted, coo.col_sorted);
}

template <>
bool SharedMemManager::CreateFromSharedMem<NDArray>(
    NDArray *nd, std::string name) {
  int ndim;
  DGLContext ctx = {kDGLCPU, 0};
  DGLDataType dtype;

  CHECK(this->Read(&ndim)) << "Invalid DGLArray file format";
  CHECK(this->Read(&dtype)) << "Invalid DGLArray file format";

  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(this->ReadArray(&shape[0], ndim)) << "Invalid DGLArray file format";
  }
  bool is_null;
  this->Read(&is_null);
  if (is_null) {
    *nd = NDArray::Empty(shape, dtype, ctx);
  } else {
    *nd = NDArray::EmptyShared(graph_name_ + name, shape, dtype, ctx, false);
  }
  return true;
}

template <>
bool SharedMemManager::CreateFromSharedMem<COOMatrix>(
    COOMatrix *coo, std::string name) {
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
bool SharedMemManager::CreateFromSharedMem<CSRMatrix>(
    CSRMatrix *csr, std::string name) {
  CreateFromSharedMem(&csr->indptr, name + "_indptr");
  CreateFromSharedMem(&csr->indices, name + "_indices");
  CreateFromSharedMem(&csr->data, name + "_data");
  strm_->Read(&csr->num_rows);
  strm_->Read(&csr->num_cols);
  strm_->Read(&csr->sorted);
  return true;
}

}  // namespace dgl
