/*!
 *  Copyright (c) 2021 Intel Corporation
 * \file array/cpu/spmm.h
 * \brief SPMM CPU kernel function header.
 * \author Sanchit Misra <sanchit.misra@intel.com>,
 *         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
 *         Vasimuddin Md <vasimuddin.md@intel.com>,
 *         Sasikanth Avancha <sasikanth.avancha@intel.com>
 */
#ifndef DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_
#define DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dmlc/logging.h>
#include <algorithm>

#if !defined(_WIN32)
#ifdef USE_AVX
#ifdef USE_LIBXSMM
#include <unistd.h>
#include <libxsmm.h>
#ifdef DEBUG
#include <x86intrin.h>
#endif
#include <omp.h>

namespace dgl {
namespace aten {
namespace cpu {

template <typename IdType, typename DType>
  struct CSRM{
    IdType num_rows;
    IdType num_cols;
    IdType *indptr;
    IdType *indices;
    DType *data;
  };

int32_t get_llc_size() {
  int32_t cache_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
  if(cache_size < 0) cache_size = DGL_CPU_LLC_SIZE;
  return cache_size;
}


template <typename IdType, typename DType>
inline void SpMMCreateBlocks(
    const CSRMatrix& csr,
    CSRM<IdType, IdType> *block_csr_array,
    IdType num_M_blocks,
    IdType num_K_blocks,
    IdType M_BLOCK_SIZE,
    IdType K_BLOCK_SIZE,
    DType *C, IdType N,
    bool use_lhs, bool use_rhs) {

  const IdType M = csr.num_rows;
  const IdType K = csr.num_cols;
  IdType* IndPtr = csr.indptr.Ptr<IdType>();
  IdType* Indices = csr.indices.Ptr<IdType>();
  IdType* Edges = csr.data.Ptr<IdType>();
  assert(IndPtr != nullptr);
  if(use_lhs)
    assert(Indices != nullptr);
  if(use_rhs)
    assert(Edges != nullptr);

  if(num_K_blocks > 1)
  {
    IdType *indptr = (IdType *)aligned_alloc(64, (M_BLOCK_SIZE + 1) * num_M_blocks * num_K_blocks * sizeof(IdType));

#pragma omp parallel
    {
      IdType *my_cur_col_id = (IdType *)aligned_alloc(64, 2 * M_BLOCK_SIZE * sizeof(IdType));

#pragma omp for
      for(IdType m = 0; m < num_M_blocks; m++)
      {
        IdType M_start = m * M_BLOCK_SIZE;
        IdType M_end = (m + 1) * M_BLOCK_SIZE;
        if(M_end > M) M_end = M;
        IdType nnz = IndPtr[M_end] - IndPtr[M_start];

        IdType cur_indices_id = 0;
        IdType *indices, *edges;
        if(use_lhs)
          indices = (IdType *)aligned_alloc(64, nnz * sizeof(IdType));
        if(use_rhs)
          edges = (IdType *)aligned_alloc(64, nnz * sizeof(IdType));

        for(IdType i = M_start; i < M_end; i++)
        {
          my_cur_col_id[(i - M_start) * 2] = IndPtr[i];
          my_cur_col_id[(i - M_start) * 2 + 1] = IndPtr[i + 1];
        }
        for(IdType k = 0; k < num_K_blocks; k++)
        {
          IdType K_start = k * K_BLOCK_SIZE;
          IdType K_end = (k + 1) * K_BLOCK_SIZE;
          if(K_end > K) K_end = K;
          CSRM<IdType, IdType> cur_csr;
          cur_csr.num_rows = M_end - M_start;
          cur_csr.num_cols = K_end - K_start;
          // Create csr_ij
          IdType *cur_csr_indptr = indptr + (m * num_K_blocks + k) * (M_BLOCK_SIZE + 1);
          IdType *cur_csr_indices = nullptr, *cur_csr_edges = nullptr;
          if(use_lhs)
            cur_csr_indices = indices + cur_indices_id;
          if(use_rhs)
            cur_csr_edges = edges + cur_indices_id;
          IdType cur_nnz = 0;
          for(IdType i = M_start; i < M_end; i++)
          {
            const IdType row_start = my_cur_col_id[(i - M_start) * 2];
            const IdType row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
            cur_csr_indptr[i - M_start] = cur_nnz;
            IdType eid;
            for(eid = row_start; eid < row_end; eid++)
            {
              const IdType src = Indices[eid];
              const IdType edge = Edges[eid];
              if(src >= K_end)
              {
                break;
              }
              assert(cur_indices_id + cur_nnz < nnz);
              if(use_lhs)
                cur_csr_indices[cur_nnz] = src;
              if(use_rhs)
                cur_csr_edges[cur_nnz] = edge;
              cur_nnz++;
            }
            my_cur_col_id[(i - M_start) * 2] = eid;
          }
          cur_csr_indptr[cur_csr.num_rows] = cur_nnz;
          cur_indices_id += cur_nnz;
          cur_csr.indptr = cur_csr_indptr;
          if(use_lhs)
            cur_csr.indices = cur_csr_indices;
          if(use_rhs)
            cur_csr.data = cur_csr_edges;
          block_csr_array[m * num_K_blocks + k] = cur_csr;

        }
        assert(nnz == cur_indices_id);
      }
      free(my_cur_col_id);
    }
  }
  else
  {
#pragma omp for
    for(IdType m = 0; m < num_M_blocks; m++)
    {
      IdType M_start = m * M_BLOCK_SIZE;
      IdType M_end = (m + 1) * M_BLOCK_SIZE;
      if(M_end > M) M_end = M;

      CSRM<IdType, IdType> cur_csr;
      cur_csr.num_rows = M_end - M_start;
      cur_csr.num_cols = K;
      cur_csr.indptr = IndPtr + M_start;
      cur_csr.indices = Indices;
      cur_csr.data = Edges;

      block_csr_array[m] = cur_csr;
    }
  }
}


template <typename IdType, typename DType, typename Op>
inline libxsmm_meltwfunction_opreduce_vecs_idx SpMMCreateLibxsmmKernel(
    bool has_idx,
    IdType N,
    int32_t _ld,
    libxsmm_meltw_opreduce_vecs_flags redop_flag,
    bool is_cmp) {

  libxsmm_meltw_opreduce_vecs_flags opredop_flags;
  if(std::is_same<Op, op::Add<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD;
  }
  else if(std::is_same<Op, op::Sub<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB;
  }
  else if(std::is_same<Op, op::Mul<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL;
  }
  else if(std::is_same<Op, op::Div<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV;
  }
  else if(std::is_same<Op, op::CopyLhs<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  }
  else if(std::is_same<Op, op::CopyRhs<DType>>::value)
  {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  }
  if(std::is_same<Op, op::CopyLhs<DType>>::value)
  {
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);
  }
  else if(std::is_same<Op, op::CopyRhs<DType>>::value)
  {
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX);
    if(!has_idx)
    {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX);
    }
  }
  else
  {
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);
    if(has_idx)
    {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC);
    }
    else
    {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC);
    }
  }
  opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | redop_flag);
  if(is_cmp)
  {
    if(Op::use_lhs)
    {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0);
    }
    if(Op::use_rhs)
    {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1);
    }
  }
  libxsmm_meltwfunction_opreduce_vecs_idx kernel=nullptr;
  if(std::is_same<DType, uint16_t>::value)
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(N, &_ld, &_ld, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);
  else if(std::is_same<DType, DType>::value)
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(N, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);
  if(kernel == nullptr)
  {
    LOG(FATAL) << "libxsmm Op-redop kernel is nullptr! Bailing...\n";
  }
  return kernel;
}

template <typename IdType, typename DType>
inline void SpMMBlockwiseOpSum(
    CSRM<IdType, IdType> *block_csr_array,
    DType *B, DType *E, DType *C, bool has_idx, IdType N,
    IdType num_M_blocks, IdType num_K_blocks, IdType M_BLOCK_SIZE,
    libxsmm_meltwfunction_opreduce_vecs_idx kernel) {

  DType (*in_matrix1)[N] = (DType (*)[N])B;
  DType (*in_matrix2)[N] = (DType (*)[N])E;
  DType (*output)[N] = (DType (*)[N])C;
#pragma omp parallel
  {
    for(IdType k = 0; k < num_K_blocks; k++)
    {
#pragma omp for schedule(dynamic)
      for(IdType m = 0; m < num_M_blocks; m++)
      {
        CSRM<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];

        int32_t cur_M = cur_csr.num_rows;

        IdType M_start = m * M_BLOCK_SIZE;
        for(IdType i = 0; i < cur_M; i++)
        {
          const IdType row_start = cur_csr.indptr[i];
          const IdType row_end   = cur_csr.indptr[i + 1];
          IdType dst = i + M_start;

          libxsmm_meltw_opreduce_vecs_idx_param params;
          params.n = row_end - row_start;
          params.indices = &cur_csr.indices[row_start];
          params.in_matrix = in_matrix1;
          params.out_vec = &output[dst][0];
          params.scale_vals = nullptr;
          if(has_idx)
          {
            params.in_matrix2 = in_matrix2;
            params.indices2 = &cur_csr.data[row_start];
          }
          else
          {
            params.in_matrix2 = &in_matrix2[row_start];
          }
          kernel(&params);
        }
      }
    }
  }
}

template <typename IdType, typename DType, typename Op, typename Cmp>
inline void SpMMBlockwiseOpCmp(
    CSRM<IdType, IdType> *block_csr_array,
    DType *B, DType *E, DType *C, IdType *argB, IdType *argE,
    bool has_idx, IdType N,
    IdType num_M_blocks, IdType num_K_blocks, IdType M_BLOCK_SIZE,
    libxsmm_meltwfunction_opreduce_vecs_idx kernel) {

  DType (*in_matrix1)[N] = (DType (*)[N])B;
  DType (*in_matrix2)[N] = (DType (*)[N])E;
  DType (*output)[N] = (DType (*)[N])C;
  IdType (*out_matrix1)[N] = (IdType (*)[N])argB;
  IdType (*out_matrix2)[N] = (IdType (*)[N])argE;

#pragma omp parallel
  {
    for(IdType k = 0; k < num_K_blocks; k++)
    {
#pragma omp for schedule(dynamic)
      for(IdType m = 0; m < num_M_blocks; m++)
      {
        CSRM<IdType, IdType> cur_csr = block_csr_array[m * num_K_blocks + k];

        int32_t cur_M = cur_csr.num_rows;

        IdType M_start = m * M_BLOCK_SIZE;
        for(IdType i = 0; i < cur_M; i++)
        {
          const IdType row_start = cur_csr.indptr[i];
          const IdType row_end   = cur_csr.indptr[i + 1];
          IdType dst = i + M_start;

          libxsmm_meltw_opreduce_vecs_idx_param params;
          params.n = row_end - row_start;
          params.indices = &cur_csr.indices[row_start];
          params.in_matrix = in_matrix1;
          params.out_vec = &output[dst][0];
          params.argop_off_vec_0 = &out_matrix1[dst][0];
          params.argop_off_vec_1 = &out_matrix2[dst][0];
          params.scale_vals = nullptr;
          if(has_idx)
          {
            params.in_matrix2 = in_matrix2;
            params.indices2 = &cur_csr.data[row_start];
          }
          else
          {
            params.in_matrix2 = &in_matrix2[row_start];
          }
          kernel(&params);
        }
      }
    }
  }
}

template <typename IdType>
inline void SpMMFreeBlocks(
    CSRM<IdType, IdType> *block_csr_array,
    IdType num_M_blocks, IdType num_K_blocks,
    bool use_lhs, bool use_rhs) {

  if(num_K_blocks > 1)
  {
    for(int m = 0; m < num_M_blocks; m++)
    {
      if(use_lhs)
        free(block_csr_array[m * num_K_blocks].indices);
      if(use_rhs)
        free(block_csr_array[m * num_K_blocks].data);
    }
    free(block_csr_array[0].indptr);
  }
  free(block_csr_array);
}

template <typename IdType, typename DType, typename Op, typename Redop>
void SpMMRedopCsrOpt(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out,
    NDArray argu, NDArray arge) {

  int32_t LLC_SIZE = get_llc_size();

#ifdef DEBUG
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif

  const bool has_idx = !IsNullArray(csr.data);

  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();
  DType* E = efeat.Ptr<DType>();
  IdType *argB, *argE;
  if(std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value)
  {
    argB = argu.Ptr<IdType>();
    argE = arge.Ptr<IdType>();
  }

  int nthreads = omp_get_max_threads();
  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  IdType* IndPtr = csr.indptr.Ptr<IdType>();
  assert(IndPtr != nullptr);
  int total_nnz = IndPtr[M];
  if(M <= 0 || K <= 0 || N <= 0 || total_nnz <= 0) return;

  float avgDegree = total_nnz * 1.0 / M;
  float nnz_prob = avgDegree / K;

  IdType K_BLOCK_SIZE = LLC_SIZE / (N * sizeof(DType) * nnz_prob * 500);
  IdType M_BLOCK_SIZE = M / (nthreads * 20);
  if (M_BLOCK_SIZE == 0) M_BLOCK_SIZE = 1;
  if (K_BLOCK_SIZE == 0) K_BLOCK_SIZE = 1;

  IdType num_M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
  IdType num_K_blocks = (K + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;

  CSRM<IdType, IdType> *block_csr_array = (CSRM<IdType, IdType> *)aligned_alloc(64, sizeof(CSRM<IdType, IdType>) * num_M_blocks * num_K_blocks);

#ifdef DEBUG
  endTick = __rdtsc();
  if(std::is_same<Redop, op::Max<DType>>::value)
  {
    LOG(INFO) << "Redop = Max";
  }
  else if(std::is_same<Redop, op::Min<DType>>::value)
  {
    LOG(INFO) << "Redop = Min";
  }
  else if(std::is_same<Redop, op::Add<DType>>::value)
  {
    LOG(INFO) << "Redop = Add";
  }
  LOG(INFO) << "nthreads = " << nthreads << ", LLC_SIZE = " << LLC_SIZE;
  LOG(INFO) << "M = " << M << ", K = " << K << ", N = " << N << ", use_lhs = " << Op::use_lhs << ", use_rhs = " << Op::use_rhs;
  LOG(INFO) << "total_nnz = " << total_nnz << ", avgDegree = " << avgDegree;
  LOG(INFO) << "has_idx = " << has_idx;
  LOG(INFO) << "nnz_prob = " << nnz_prob;
  LOG(INFO) << "K_BLOCK_SIZE = " << K_BLOCK_SIZE << ", M_BLOCK_SIZE = " << M_BLOCK_SIZE;
  LOG(INFO) << "num_K_blocks = " << num_K_blocks << ", num_M_blocks = " << num_M_blocks;
  LOG(INFO) << "stage0 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif

  SpMMCreateBlocks(csr, block_csr_array, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, K_BLOCK_SIZE, C, N, Op::use_lhs, Op::use_rhs);

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage1 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif

  int _ld = N;
  libxsmm_meltwfunction_opreduce_vecs_idx kernel = nullptr;
  if(std::is_same<Redop, op::Max<DType>>::value)
  {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX, true);
  }
  else if(std::is_same<Redop, op::Min<DType>>::value)
  {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN, true);
  }
  else if(std::is_same<Redop, op::Add<DType>>::value)
  {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM, false);
  }

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage2 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif

  if(std::is_same<Redop, op::Max<DType>>::value || std::is_same<Redop, op::Min<DType>>::value)
  {
    SpMMBlockwiseOpCmp<IdType, DType, Op, Redop>(block_csr_array, B, E, C, argB, argE, has_idx, N, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, kernel);
  }
  else
  {
    SpMMBlockwiseOpSum(block_csr_array, B, E, C, has_idx, N, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, kernel);
  }

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage3 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif

  SpMMFreeBlocks(block_csr_array, num_M_blocks, num_K_blocks, Op::use_lhs, Op::use_rhs);

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage4 ticks = " << (endTick - startTick);
#endif
}

template <typename IdType, typename DType, typename Op>
void SpMMSumCsrOpt(
	const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  
  NDArray dummy;
  SpMMRedopCsrOpt<IdType, DType, Op, op::Add<DType>>(bcast, csr, ufeat, efeat, out, dummy, dummy);
}
 
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsrOpt(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out,
    NDArray argu, NDArray arge) {

  SpMMRedopCsrOpt<IdType, DType, Op, Cmp>(bcast, csr, ufeat, efeat, out, argu, arge);
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // USE_LIBXSMM
#endif  // USE_AVX
#endif  // _WIN32

#endif  // DGL_ARRAY_CPU_SPMM_H_
