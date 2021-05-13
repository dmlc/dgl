/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm.h
 * \brief SPMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SPMM_H_
#define DGL_ARRAY_CPU_SPMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <algorithm>
#include <limits>
#include <memory>
#include "spmm_binary_ops.h"
#if !defined(_WIN32)
#ifdef USE_AVX
#include <cpuid.h>
#include <libxsmm.h>
#endif  // USE_AVX
#endif  // _WIN32

#include <iostream>
#include <iomanip>
#include <x86intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>

#define SPMM_LOG_INFO 0

namespace dgl {
namespace aten {
namespace cpu {

#if !defined(_WIN32)
#ifdef USE_AVX
template <typename IdType, typename DType>
  struct CSRM{
    IdType num_rows;
    IdType num_cols;
    IdType *indptr;
    IdType *indices;
    DType *data;
  };

int32_t get_llc_size() {

  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;

  __get_cpuid_count(4, 3, &eax, &ebx, &ecx, &edx);
  unsigned int cache_size = ((ebx >> 22) + 1) * (((ebx >> 12) & 0x3ff) + 1) * ((ebx & 0xfff) + 1) * (ecx + 1);
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
  assert(IndPtr != NULL);
  if(use_lhs)
    assert(Indices != NULL);
  if(use_rhs)
    assert(Edges != NULL);

  if(num_K_blocks > 1)
  {
    IdType *indptr = (IdType *)_mm_malloc((M_BLOCK_SIZE + 1) * num_M_blocks * num_K_blocks * sizeof(IdType), 64);

#pragma omp parallel
    {
      IdType *my_cur_col_id = (IdType *)_mm_malloc(2 * M_BLOCK_SIZE * sizeof(IdType), 64);

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
          indices = (IdType *)_mm_malloc(nnz * sizeof(IdType), 64);
        if(use_rhs)
          edges = (IdType *)_mm_malloc(nnz * sizeof(IdType), 64);

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
          IdType *cur_csr_indices = NULL, *cur_csr_edges = NULL;
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
      _mm_free(my_cur_col_id);
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
  libxsmm_meltwfunction_opreduce_vecs_idx kernel=NULL;
  if(std::is_same<DType, uint16_t>::value)
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(N, &_ld, &_ld, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);
  else if(std::is_same<DType, DType>::value)
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(N, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);
  if(kernel == NULL)
  {
    printf("Op-redop kernel is NULL! Bailing...\n");
    exit(-1);
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
#if 0
#pragma omp for
    for(IdType m = 0; m < num_M_blocks; m++)
    {
      memset((void *)(&output[m * M_BLOCK_SIZE][0]), 0, block_csr_array[m * num_K_blocks].num_rows * N * sizeof(DType));
    }
#pragma omp barrier
#endif
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
          params.scale_vals = NULL;
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
#pragma omp for
    for(IdType m = 0; m < num_M_blocks; m++)
    {
      int64_t num_rows = block_csr_array[m * num_K_blocks].num_rows; 
      for(int64_t i = 0; i < num_rows; i++)
      {
        for(int64_t j = 0; j < N; j++)
          output[m * M_BLOCK_SIZE + i][j] = Cmp::zero;
      }
#if 0
      if(Op::use_lhs)
        memset((void *)(&out_matrix1[m * M_BLOCK_SIZE][0]), 0, size * sizeof(IdType));
      if(Op::use_rhs)
        memset((void *)(&out_matrix2[m * M_BLOCK_SIZE][0]), 0, size * sizeof(IdType));
#endif
    }
#pragma omp barrier
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
          params.scale_vals = NULL;
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
        _mm_free(block_csr_array[m * num_K_blocks].indices);
      if(use_rhs)
        _mm_free(block_csr_array[m * num_K_blocks].data);
    }
    _mm_free(block_csr_array[0].indptr);
  }
  _mm_free(block_csr_array);
}

template <typename IdType, typename DType, typename Op>
void SpMMSumCsrOpt(
	const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {

  int32_t LLC_SIZE = get_llc_size();

#if SPMM_LOG_INFO
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif

  const bool has_idx = !IsNullArray(csr.data);

  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();
  DType* E = efeat.Ptr<DType>();

  int nthreads = omp_get_max_threads();
  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  IdType* IndPtr = csr.indptr.Ptr<IdType>();
  assert(IndPtr != NULL);
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

  CSRM<IdType, IdType> *block_csr_array = (CSRM<IdType, IdType> *)_mm_malloc(sizeof(CSRM<IdType, IdType>) * num_M_blocks * num_K_blocks, 64);
#if SPMM_LOG_INFO
  printf("nthreads = %d\n", nthreads);
  printf("M = %d, K = %d, N = %d, use_lhs = %d, use_rhs = %d\n", M, K, N, Op::use_lhs, Op::use_rhs);
  printf("total_nnz = %d, avgDegree = %f\n", total_nnz, avgDegree);
  printf("has_idx = %d\n", has_idx);
  printf("nnz_prob = %lf\n", nnz_prob);
  printf("K_BLOCK_SIZE = %d, M_BLOCK_SIZE = %d\n", K_BLOCK_SIZE, M_BLOCK_SIZE);
  printf("num_K_blocks = %d, num_M_blocks = %d\n", num_K_blocks, num_M_blocks);
#endif

#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage0: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
  SpMMCreateBlocks(csr, block_csr_array, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, K_BLOCK_SIZE, C, N, Op::use_lhs, Op::use_rhs);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage1: %ld ticks\n", endTick - startTick);
#endif

#if SPMM_LOG_INFO
  startTick = __rdtsc();
#endif
  int _ld = N;
  libxsmm_meltwfunction_opreduce_vecs_idx kernel = NULL;
  kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM, false);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage2: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
  SpMMBlockwiseOpSum(block_csr_array, B, E, C, has_idx, N, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, kernel);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
#endif
#if SPMM_LOG_INFO
  printf("stage3: %ld ticks\n", endTick - startTick);
  startTick = __rdtsc();
#endif

  SpMMFreeBlocks(block_csr_array, num_M_blocks, num_K_blocks, Op::use_lhs, Op::use_rhs);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage4: %ld ticks\n", endTick - startTick);
#endif
}
 
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsrOpt(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out,
    NDArray argu, NDArray arge) {

  int32_t LLC_SIZE = get_llc_size();

#if SPMM_LOG_INFO
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif

  const bool has_idx = !IsNullArray(csr.data);

  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();
  DType* E = efeat.Ptr<DType>();
  IdType *argB = argu.Ptr<IdType>();
  IdType *argE = arge.Ptr<IdType>();

  int nthreads = omp_get_max_threads();
  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  IdType* IndPtr = csr.indptr.Ptr<IdType>();
  assert(IndPtr != NULL);
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

  CSRM<IdType, IdType> *block_csr_array = (CSRM<IdType, IdType> *)_mm_malloc(sizeof(CSRM<IdType, IdType>) * num_M_blocks * num_K_blocks, 64);
#if SPMM_LOG_INFO
  printf("nthreads = %d\n", nthreads);
  printf("M = %d, total_nnz = %d, avgDegree = %f\n", M, total_nnz, avgDegree);
  printf("has_idx = %d\n", has_idx);
  printf("nnz_prob = %lf\n", nnz_prob);
  printf("K_BLOCK_SIZE = %d, M_BLOCK_SIZE = %d\n", K_BLOCK_SIZE, M_BLOCK_SIZE);
  printf("num_K_blocks = %d, num_M_blocks = %d\n", num_K_blocks, num_M_blocks);
#endif

#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage0: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
  SpMMCreateBlocks(csr, block_csr_array, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, K_BLOCK_SIZE, C, N, Op::use_lhs, Op::use_rhs);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage1: %ld ticks\n", endTick - startTick);
#endif

#if SPMM_LOG_INFO
  startTick = __rdtsc();
#endif
  int _ld = N;
  libxsmm_meltwfunction_opreduce_vecs_idx kernel = NULL;
  if(std::is_same<Cmp, op::Max<DType>>::value)
  {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX, true);
  }
  else
  {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(has_idx, N, _ld, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN, true);
  }
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage2: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
  SpMMBlockwiseOpCmp<IdType, DType, Op, Cmp>(block_csr_array, B, E, C, argB, argE, has_idx, N, num_M_blocks, num_K_blocks, M_BLOCK_SIZE, kernel);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
#endif
#if SPMM_LOG_INFO
  printf("stage3: %ld ticks\n", endTick - startTick);
  startTick = __rdtsc();
#endif

  SpMMFreeBlocks(block_csr_array, num_M_blocks, num_K_blocks, Op::use_lhs, Op::use_rhs);
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage4: %ld ticks\n", endTick - startTick);
#endif
}
#endif
#endif

/*!
 * \brief CPU kernel of SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat,
                NDArray efeat, NDArray out) {

  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  assert(indptr != NULL);
  if(Op::use_lhs)
  {
      assert(indices != NULL);
      assert(X != NULL);
  }
  if(Op::use_rhs)
  {
      if(has_idx)
          assert(edges != NULL);
      assert(W != NULL);
  }
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
  assert(O != NULL);

#if !defined(_WIN32)
#ifdef USE_AVX
  
  const bool sorted = csr.sorted;
  if(!sorted)
  {
    printf("CSR is not sorted!\n");
  }
  bool special_condition = (!sorted) || bcast.use_bcast || (Op::use_lhs && (dim != lhs_dim)) || (Op::use_rhs && (dim != rhs_dim));
  if(!special_condition)
  {
      SpMMSumCsrOpt<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
  } else {
#endif  // USE_AVX
#endif  // _WIN32

#pragma omp parallel for
    for (IdType rid = 0; rid < csr.num_rows; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      DType* out_off = O + rid * dim;
      std::fill(out_off, out_off + dim, 0);
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        for (int64_t k = 0; k < dim; ++k) {
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
            Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
          const DType* rhs_off =
            Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
          out_off[k] += Op::Call(lhs_off, rhs_off);
        }
      }
    }
#if !defined(_WIN32)
#ifdef USE_AVX
  }
#endif  // USE_AVX
#endif  // _WIN32
}

/*!
 * \brief CPU kernel of SpMM-Min/Max on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer. \param arge Arg-Min/Max on edges. which refers the source node
 * indices correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer. \note It uses node parallel strategy, different threads are
 * responsible for the computation of different nodes. \note The result will
 * contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat,
                NDArray efeat, NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges =
    has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs ? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs ? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs ? static_cast<IdType*>(arge->data) : nullptr;
#if !defined(_WIN32)
#ifdef USE_AVX
  
  const bool sorted = csr.sorted;
  if(!sorted)
  {
    printf("CSR is not sorted!\n");
  }
  bool special_condition = (!sorted) || bcast.use_bcast || (Op::use_lhs && (dim != lhs_dim)) || (Op::use_rhs && (dim != rhs_dim));
  if(!special_condition)
  {
      SpMMCmpCsrOpt<IdType, DType, Op, Cmp>(bcast, csr, ufeat, efeat, out, argu, arge);
  } else {
#endif  // USE_AVX
#endif  // _WIN32

#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    std::fill(out_off, out_off + dim, Cmp::zero);
    if (Op::use_lhs) std::fill(argx_off, argx_off + dim, 0);
    if (Op::use_rhs) std::fill(argw_off, argw_off + dim, 0);
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx ? edges[j] : j;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off =
          Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
        const DType* rhs_off =
          Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(out_off[k], val)) {
          out_off[k] = val;
          if (Op::use_lhs) argx_off[k] = cid;
          if (Op::use_rhs) argw_off[k] = eid;
        }
      }
    }
  }
#if !defined(_WIN32)
#ifdef USE_AVX
  }
#endif  // USE_AVX
#endif  // _WIN32
}


/*!
 * \brief CPU kernel of SpMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCoo(const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat,
                NDArray efeat, NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
  assert(O != NULL);
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  memset(O, 0, out.GetSize());
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[i] : i;
    DType* out_off = O + cid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off =
        Op::use_lhs ? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off =
        Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
      if (val != 0) {
#pragma omp atomic
        out_off[k] += val;
      }
    }
  }
}


/*!
 * \brief CPU kernel of SpMM-Min/Max on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer. \param arge Arg-Min/Max on edges. which refers the source node
 * indices correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer. \note it uses node parallel strategy, different threads are
 * responsible for the computation of different nodes. To avoid possible data
 * hazard, we use atomic operators in the reduction phase. \note The result will
 * contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCoo(const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat,
                NDArray efeat, NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges =
    has_idx ? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs ? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs ? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs ? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  std::fill(O, O + out.NumElements(), Cmp::zero);
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[i] : i;
    DType* out_off = O + cid * dim;
    IdType* argx_off = Op::use_lhs ? argX + cid * dim : nullptr;
    IdType* argw_off = Op::use_rhs ? argW + cid * dim : nullptr;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off =
        Op::use_lhs ? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off =
        Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp critical
      if (Cmp::Call(out_off[k], val)) {
        out_off[k] = val;
        if (Op::use_lhs) argx_off[k] = rid;
        if (Op::use_rhs) argw_off[k] = eid;
      }
    }
  }
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_H_
