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
#include "intel/cpu_support.h"
#endif  // USE_AVX
#endif  // _WIN32

#if 1
#include <iostream>
#include <iomanip>
#include <x86intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
// #include <mkl_spblas.h>
// #include <mkl.h>
#include <omp.h>
#include <vector>

 #ifdef VTUNE_ANALYSIS
 #include <ittnotify.h> 
 #endif

 double procf = 2.7*1e9;

 #define USE_LIBXSMM 1

 #if USE_LIBXSMM
 #include <libxsmm.h>
 #endif

 #define SPMM_LOG_INFO 0
 #define PRINT_HIST 0
 #define RANDOM 1
 #define MKL 0
 #define ANA 0
 #define QSORT 0
 #define ENABLE_PREFETCH 1
 #if ENABLE_PREFETCH
 #define MM_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
 #else
 #define MM_PREFETCH(addr) 
 #endif
#endif

namespace dgl {
namespace aten {
namespace cpu {

#if 1

template <typename T>
struct NDArray1{
    T *data;

    T *Ptr()
    {
        return data;
    }
};

template <typename IdType, typename DType>
  struct CSRM{
      IdType num_rows;
      IdType num_cols;
      NDArray1<IdType> indptr, indices;
  };

template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out)
{
#define LLC_SIZE 40000000

#if SPMM_LOG_INFO
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif

  IdType* IndPtr = csr.indptr.Ptr<IdType>();
  IdType* Indices = csr.indices.Ptr<IdType>();

  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();

  int nthreads = omp_get_max_threads();
  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  int total_nnz = IndPtr[M];
  float avgDegree = total_nnz * 1.0 / M;
  float nnz_prob = avgDegree / K;

  IdType K_BLOCK_SIZE = LLC_SIZE / (N * sizeof(DType) * nnz_prob * 500);
  IdType M_BLOCK_SIZE = M / (nthreads * 20);
  if (M_BLOCK_SIZE == 0) M_BLOCK_SIZE = 1;
  if (K_BLOCK_SIZE == 0) K_BLOCK_SIZE = 1;

  IdType num_M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
  IdType num_K_blocks = (K + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;

  CSRM<IdType, DType> *block_csr_array = (CSRM<IdType, DType> *)_mm_malloc(sizeof(CSRM<IdType, DType>) * num_M_blocks * num_K_blocks, 64);
#if SPMM_LOG_INFO
  printf("nthreads = %d\n", nthreads);
  printf("M = %d, total_nnz = %d, avgDegree = %f\n", M, total_nnz, avgDegree);
  printf("nnz_prob = %lf\n", nnz_prob);
  printf("K_BLOCK_SIZE = %d, M_BLOCK_SIZE = %d\n", K_BLOCK_SIZE, M_BLOCK_SIZE);
  printf("num_K_blocks = %d, num_M_blocks = %d\n", num_K_blocks, num_M_blocks);
#endif
  #if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage0: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
  if(num_K_blocks > 1)
  {
    IdType *indptr = (IdType *)_mm_malloc((M_BLOCK_SIZE + 1) * num_M_blocks * num_K_blocks * sizeof(IdType), 64);

#pragma omp parallel
    {
      IdType *my_cur_col_id = (IdType *)_mm_malloc(2 * M_BLOCK_SIZE * sizeof(IdType), 64);
      uint64_t tst = __rdtsc();

      int tid = omp_get_thread_num();
#pragma omp for
      for(IdType m = 0; m < num_M_blocks; m++)
      {
        IdType M_start = m * M_BLOCK_SIZE;
        IdType M_end = (m + 1) * M_BLOCK_SIZE;
        if(M_end > M) M_end = M;
        IdType nnz = IndPtr[M_end] - IndPtr[M_start];

        IdType cur_indices_id = 0;
        IdType *indices = (IdType *)_mm_malloc(nnz * sizeof(IdType), 64);

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
          CSRM<IdType, DType> cur_csr;
          cur_csr.num_rows = M_end - M_start;
          cur_csr.num_cols = K_end - K_start;
          // Create csr_ij
          IdType *cur_csr_indptr = indptr + (m * num_K_blocks + k) * (M_BLOCK_SIZE + 1);
          IdType *cur_csr_indices = indices + cur_indices_id;
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
              if(src >= K_end)
              {
                break;
              }
              if(cur_indices_id + cur_nnz >= nnz)
              {
                printf("Error! cur_indices_id + cur_nnz = %ld, nnz = %ld\n", cur_indices_id + cur_nnz, nnz);
                exit(0);
              }
              cur_csr_indices[cur_nnz] = src;
              cur_nnz++;
            }
            my_cur_col_id[(i - M_start) * 2] = eid;
          }
          cur_csr_indptr[cur_csr.num_rows] = cur_nnz;
          cur_indices_id += cur_nnz;
          cur_csr.indptr.data = cur_csr_indptr; // TODO: modify this
          cur_csr.indices.data = cur_csr_indices; // TODO: modify this
          block_csr_array[m * num_K_blocks + k] = cur_csr;

        }
        if(nnz != cur_indices_id)
        {
          printf("cur_indices_id = %ld, expected = %ld\n", cur_indices_id, nnz);
          exit(0);
        }
      }
      _mm_free(my_cur_col_id);
      uint64_t tend = __rdtsc();
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
      IdType nnz = IndPtr[M_end] - IndPtr[M_start];

      CSRM<IdType, DType> cur_csr;
      cur_csr.num_rows = M_end - M_start;
      cur_csr.num_cols = K;
      cur_csr.indptr.data = IndPtr + M_start; // TODO: modify this
      cur_csr.indices.data = Indices; // TODO: modify this

      block_csr_array[m] = cur_csr;
    }
  }
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage1: %ld ticks\n", endTick - startTick);
#endif

#ifdef VTUNE_ANALYSIS
  __itt_resume();
#endif

#if SPMM_LOG_INFO
  startTick = __rdtsc();
#endif
  DType (*input)[N] = (DType (*)[N])B;
  DType (*output)[N] = (DType (*)[N])C;
#if USE_LIBXSMM
  int _ld = N;
  libxsmm_meltw_opreduce_vecs_flags opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY_REDOP_SUM;
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
#endif
#if SPMM_LOG_INFO
  endTick = __rdtsc();
  printf("stage2: %ld ticks\n", endTick - startTick);

  startTick = __rdtsc();
#endif
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    uint64_t tst = __rdtsc();
    for(IdType k = 0; k < num_K_blocks; k++)
    {
#pragma omp for schedule(dynamic)
      for(IdType m = 0; m < num_M_blocks; m++)
      {
        CSRM<IdType, DType> cur_csr = block_csr_array[m * num_K_blocks + k];

        int32_t cur_M = cur_csr.num_rows;
        int32_t cur_K = cur_csr.num_cols;

        IdType M_start = m * M_BLOCK_SIZE;
        for(IdType i = 0; i < cur_M; i++)
        {
          const IdType row_start = cur_csr.indptr.Ptr()[i];
          const IdType row_end   = cur_csr.indptr.Ptr()[i + 1];
          IdType dst = i + M_start;

          libxsmm_meltw_opreduce_vecs_idx_param params;
          params.n = row_end - row_start;
          params.indices = &((cur_csr.indices.Ptr())[row_start]);
          params.in_matrix = input;
          params.in_vec = NULL;
          params.out_vec = &output[dst][0];
          params.scale_vals = NULL;
          kernel(&params);

        }
      }
    }
    uint64_t tend = __rdtsc();
  }
  #if SPMM_LOG_INFO
    endTick = __rdtsc();
  #endif
  #ifdef VTUNE_ANALYSIS
    __itt_pause();
  #endif
  #if SPMM_LOG_INFO
    printf("stage3: %ld ticks\n", endTick - startTick);
    startTick = __rdtsc();
  #endif

    if(num_K_blocks > 1)
    {
      for(int m = 0; m < num_M_blocks; m++)
      {
        _mm_free(block_csr_array[m * num_K_blocks].indices.data);
      }
      _mm_free(block_csr_array[0].indptr.data);
    }
    _mm_free(block_csr_array);
  #if SPMM_LOG_INFO
    endTick = __rdtsc();
    printf("stage4: %ld ticks\n", endTick - startTick);
  #endif
  #undef LLC_SIZE
}
  


#else
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
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
#if !defined(_WIN32)
#ifdef USE_AVX
  //printf("#if\n");
  typedef dgl::ElemWiseAddUpdate<Op> ElemWiseUpd;
  /* Prepare an assembler kernel */
  static std::unique_ptr<ElemWiseUpd> asm_kernel_ptr(
    (dgl::IntelKernel<>::IsEnabled()) ? new ElemWiseUpd() : nullptr);
  /* Distribute the kernel among OMP threads */
  ElemWiseUpd* cpu_spec = (asm_kernel_ptr && asm_kernel_ptr->applicable())
                            ? asm_kernel_ptr.get()
                            : nullptr;
  //printf("cpu_spec = %ld, dim = %d\n", cpu_spec, dim);
  if (cpu_spec && dim > 16 && !bcast.use_bcast) {
      //printf("xybak\n");
#pragma omp parallel for
    for (IdType rid = 0; rid < csr.num_rows; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      DType* out_off = O + rid * dim;
      std::fill(out_off, out_off + dim, 0);
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        cpu_spec->run(out_off, X + cid * lhs_dim, W + eid * rhs_dim, dim);
      }
    }
  } else {
#endif  // USE_AVX
#endif  // _WIN32
      //printf("no xybak\n");

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

#endif

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
