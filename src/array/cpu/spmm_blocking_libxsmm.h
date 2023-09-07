/**
 *  Copyright (c) 2021 Intel Corporation
 * @file array/cpu/spmm.h
 * @brief SPMM CPU kernel function header.
 * @author Sanchit Misra <sanchit.misra@intel.com>,
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
#ifdef USE_LIBXSMM
#include <libxsmm_source.h>
#include <unistd.h>
#ifdef DEBUG
#include <x86intrin.h>
#endif  // DEBUG
#include <dmlc/omp.h>

#define NUM_BLOCKS_PER_THREAD 20
#define BLOCKING_HEURISTIC_PARAM 500

namespace dgl {
namespace aten {
namespace cpu {

template <typename IdType, typename DType>
struct CSRMatrixInternal {
  IdType num_rows;
  IdType num_cols;
  IdType *indptr;
  IdType *indices;
  DType *data;
};

int32_t GetLLCSize() {
#ifdef _SC_LEVEL3_CACHE_SIZE
  int32_t cache_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
  if (cache_size < 0) cache_size = DGL_CPU_LLC_SIZE;
#else
  int32_t cache_size = DGL_CPU_LLC_SIZE;
#endif
  return cache_size;
}

/**
 * @brief Tile the CSR matrix to roughly make sure that the column tiles and
 *        corresponding neighbor features fit into LLC and the row tiles
 *        are assigned to OMP threads.
 * @param csr The Csr matrix.
 * @param block_csr_array The array containing csr matrices of all blocks.
 * @param num_M_blocks Number of blocks to create along the rows of adjacency
 *        matrix.
 * @param num_K_blocks Number of blocks to create along the columns of adjacency
 *        matrix.
 * @param M_block_size block size along the rows of adjacency matrix.
 * @param K_block_size block size along the columns of adjacency matrix.
 * @param use_lhs Whether to use lhs.
 * @param use_rhs Whether to use rhs.
 */
template <typename IdType>
inline void SpMMCreateBlocks(
    const CSRMatrix &csr, CSRMatrixInternal<IdType, IdType> *block_csr_array,
    IdType num_M_blocks, IdType num_K_blocks, IdType M_block_size,
    IdType K_block_size, bool use_lhs, bool use_rhs) {
  const IdType M = csr.num_rows;
  const IdType K = csr.num_cols;
  IdType *indptr = csr.indptr.Ptr<IdType>();
  IdType *indices = csr.indices.Ptr<IdType>();
  IdType *edges = csr.data.Ptr<IdType>();
  CHECK_NOTNULL(indptr);
  if (use_lhs) CHECK_NOTNULL(indices);
  if (use_rhs) CHECK_NOTNULL(edges);

  if (num_K_blocks > 1) {
    IdType *indptr_block_buf = reinterpret_cast<IdType *>(aligned_alloc(
        64, (M_block_size + 1) * num_M_blocks * num_K_blocks * sizeof(IdType)));
    IdType *indices_block_buf = nullptr;
    if (use_lhs) {
      indices_block_buf = reinterpret_cast<IdType *>(
          aligned_alloc(64, indptr[M] * sizeof(IdType)));
    }
    IdType *edges_block_buf = nullptr;
    if (use_rhs) {
      edges_block_buf = reinterpret_cast<IdType *>(
          aligned_alloc(64, indptr[M] * sizeof(IdType)));
    }

#pragma omp parallel
    {
      IdType *my_cur_col_id = reinterpret_cast<IdType *>(
          aligned_alloc(64, 2 * M_block_size * sizeof(IdType)));

#pragma omp for
      for (IdType m = 0; m < num_M_blocks; m++) {
        const IdType M_start = m * M_block_size;
        const IdType M_end = std::min((m + 1) * M_block_size, M);
        const IdType nnz = indptr[M_end] - indptr[M_start];

        IdType cur_indices_id = 0;
        IdType *my_indices_block_buf, *my_edges_block_buf;
        if (use_lhs) my_indices_block_buf = indices_block_buf + indptr[M_start];
        if (use_rhs) my_edges_block_buf = edges_block_buf + indptr[M_start];

        for (IdType i = M_start; i < M_end; i++) {
          my_cur_col_id[(i - M_start) * 2] = indptr[i];
          my_cur_col_id[(i - M_start) * 2 + 1] = indptr[i + 1];
        }
        for (IdType k = 0; k < num_K_blocks; k++) {
          const IdType K_start = k * K_block_size;
          const IdType K_end = std::min((k + 1) * K_block_size, K);
          CSRMatrixInternal<IdType, IdType> cur_csr;
          cur_csr.num_rows = M_end - M_start;
          cur_csr.num_cols = K_end - K_start;
          // Create csr_ij
          IdType *cur_csr_indptr =
              indptr_block_buf + (m * num_K_blocks + k) * (M_block_size + 1);
          IdType *cur_csr_indices = nullptr, *cur_csr_edges = nullptr;
          if (use_lhs) cur_csr_indices = my_indices_block_buf + cur_indices_id;
          if (use_rhs) cur_csr_edges = my_edges_block_buf + cur_indices_id;
          IdType cur_nnz = 0;
          for (IdType i = M_start; i < M_end; i++) {
            const IdType row_start = my_cur_col_id[(i - M_start) * 2];
            const IdType row_end = my_cur_col_id[(i - M_start) * 2 + 1];
            cur_csr_indptr[i - M_start] = cur_nnz;
            IdType eid;
            for (eid = row_start; eid < row_end; eid++) {
              const IdType src = indices[eid];
              const IdType edge = edges[eid];
              if (src >= K_end) {
                break;
              }
              CHECK_LT(cur_indices_id + cur_nnz, nnz);
              if (use_lhs) cur_csr_indices[cur_nnz] = src;
              if (use_rhs) cur_csr_edges[cur_nnz] = edge;
              cur_nnz++;
            }
            my_cur_col_id[(i - M_start) * 2] = eid;
          }
          cur_csr_indptr[cur_csr.num_rows] = cur_nnz;
          cur_indices_id += cur_nnz;
          cur_csr.indptr = cur_csr_indptr;
          if (use_lhs) cur_csr.indices = cur_csr_indices;
          if (use_rhs) cur_csr.data = cur_csr_edges;
          block_csr_array[m * num_K_blocks + k] = cur_csr;
        }
        CHECK_EQ(nnz, cur_indices_id);
      }
      free(my_cur_col_id);
    }
  } else {
    for (IdType m = 0; m < num_M_blocks; m++) {
      const IdType M_start = m * M_block_size;
      const IdType M_end = std::min((m + 1) * M_block_size, M);

      CSRMatrixInternal<IdType, IdType> cur_csr;
      cur_csr.num_rows = M_end - M_start;
      cur_csr.num_cols = K;
      cur_csr.indptr = indptr + M_start;
      cur_csr.indices = indices;
      cur_csr.data = edges;

      block_csr_array[m] = cur_csr;
    }
  }
}

/**
 * @brief Create libxsmm kernel.
 * @param has_idx For the edge features, are there indices available.
 * @param N Feature size.
 * @param redop_flag Flag specifying the reduction operation.
 * @param is_cmp Is the reduction operation a compare operation.
 * @note libxsmm_dispatch_meltw_opreduce_vecs_idx creates a JIT'ed kernel.
 *       Given a node u, the kernel performs an elementwise "Op" on the
 *       features of the neighbors and/or the edges incident on u.
 *       Subsequently, it performs an elementwise "Redop" on all such
 *       features created and stores into the feature of node u.
 *       It uses a SIMD and a cache efficient design and also provides
 *       support to enable software prefetching if needed. For IdType,
 *       it supports INT32 and INT64. For DType, it supports BF16 and FP32.
 *       It supports all the "Ops" and "Redops" supported by DGL. Once a
 *       kernel is generated by libxsmm_dispatch_meltw_opreduce_vecs_idx,
 *       it is cached for the entire duration of the execution of a program
 *       so that subsequently if the kernel is needed again, it just returns
 *       the cached copy.
 */
template <typename IdType, typename DType, typename Op>
inline libxsmm_meltwfunction_opreduce_vecs_idx SpMMCreateLibxsmmKernel(
    bool has_idx, IdType N, libxsmm_meltw_opreduce_vecs_flags redop_flag,
    bool is_cmp) {
  int _ld = N;
  libxsmm_meltw_opreduce_vecs_flags opredop_flags;
  // First, set the Op in the opredop_flags
  if (std::is_same<Op, op::Add<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD;
  } else if (std::is_same<Op, op::Sub<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB;
  } else if (std::is_same<Op, op::Mul<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL;
  } else if (std::is_same<Op, op::Div<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV;
  } else if (std::is_same<Op, op::CopyLhs<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  } else if (std::is_same<Op, op::CopyRhs<DType>>::value) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  }
  // Second, set which of lhs or rhs is considered first and second operand.
  // This is needed since libxsmm assumes that the copy operation always copies
  // the first operand. So, if we need to copy rhs, we need to set that as the
  // first operand. For rhs, we also set whether to use implicit indices or
  // provided indices.
  // TODO(Steve): fix this long line in a separate PR.
  if (std::is_same<Op, op::CopyLhs<DType>>::value) {
    opredop_flags =
        (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);  // NOLINT
  } else if (std::is_same<Op, op::CopyRhs<DType>>::value) {
    opredop_flags =
        (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX);  // NOLINT
    if (!has_idx) {
      opredop_flags =
          (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX);  // NOLINT
    }
  } else {
    opredop_flags =
        (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);  // NOLINT
    if (has_idx) {
      opredop_flags =
          (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC);  // NOLINT
    } else {
      opredop_flags =
          (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC);  // NOLINT
    }
  }
  // Third, we set the Redop in the opredop_flags
  opredop_flags =
      (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | redop_flag);
  // Fourth, in case of Cmp Redop, set whether to record argmax/argmin for
  // lhs/rhs
  if (is_cmp) {
    if (Op::use_lhs) {
      opredop_flags =
          (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0);  // NOLINT
    }
    if (Op::use_rhs) {
      opredop_flags =
          (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1);  // NOLINT
    }
  }
  libxsmm_meltwfunction_opreduce_vecs_idx kernel = nullptr;
  if (std::is_same<DType, float>::value) {
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(
        N, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32,
        opredop_flags, 0);
  } else {  // assume bf16
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(
        N, &_ld, &_ld, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
        (sizeof(IdType) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32,
        opredop_flags, 0);
  }

  if (kernel == nullptr) {
    LOG(FATAL) << "Failed to generate libxsmm kernel for the SpMM operation."
                  "To disable libxsmm, use dgl.use_libxsmm(false).";
  }
  return kernel;
}

/**
 * @brief Use libxsmm to perform SpMM-Sum on all blocks.
 * @param block_csr_array The array containing csr matrices of all blocks.
 * @param B The feature on source nodes.
 * @param E The feature on edges.
 * @param C The result feature on destination nodes.
 * @param has_idx For the edge features, are there indices available.
 * @param N Feature size.
 * @param num_M_blocks Number of blocks to create along the rows of adjacency
 *        matrix.
 * @param num_K_blocks Number of blocks to create along the columns of adjacency
 *        matrix.
 * @param M_block_size block size along the rows of adjacency matrix.
 * @param kernel The libxsmm kernel.
 */
template <typename IdType, typename DType>
inline void SpMMBlockwiseOpSum(
    CSRMatrixInternal<IdType, IdType> *block_csr_array, const DType *B,
    const DType *E, DType *C, bool has_idx, IdType N, IdType num_M_blocks,
    IdType num_K_blocks, IdType M_block_size,
    libxsmm_meltwfunction_opreduce_vecs_idx kernel) {
  const DType *in_matrix1 = B;
  const DType *in_matrix2 = E;
  DType *output = C;
#pragma omp parallel
  {
    for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
      for (IdType m = 0; m < num_M_blocks; m++) {
        CSRMatrixInternal<IdType, IdType> cur_csr =
            block_csr_array[m * num_K_blocks + k];

        const IdType M_start = m * M_block_size;
        for (IdType i = 0; i < cur_csr.num_rows; i++) {
          const IdType row_start = cur_csr.indptr[i];
          const IdType row_end = cur_csr.indptr[i + 1];
          const IdType dst = i + M_start;

          libxsmm_meltw_opreduce_vecs_idx_param params;
          params.n = row_end - row_start;
          params.indices = &cur_csr.indices[row_start];
          params.in_matrix = in_matrix1;
          params.out_vec = &output[dst * N];
          params.scale_vals = nullptr;
          if (has_idx) {
            params.in_matrix2 = in_matrix2;
            params.indices2 = &cur_csr.data[row_start];
          } else {
            params.in_matrix2 = &in_matrix2[row_start * N];
          }
          kernel(&params);
        }
      }
    }
  }
}

/**
 * @brief Use libxsmm to perform SpMM-Max/Min on all blocks.
 * @param block_csr_array The array containing csr matrices of all blocks.
 * @param B The feature on source nodes.
 * @param E The feature on edges.
 * @param C The result feature on destination nodes.
 * @param argB Arg-Min/Max on source nodes.
 * @param argE Arg-Min/Max on edges.
 * @param has_idx For the edge features, are there indices available.
 * @param N Feature size.
 * @param num_M_blocks Number of blocks to create along the rows of adjacency
 *        matrix.
 * @param num_K_blocks Number of blocks to create along the columns of adjacency
 *        matrix.
 * @param M_block_size block size along the rows of adjacency matrix.
 * @param kernel The libxsmm kernel.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
inline void SpMMBlockwiseOpCmp(
    CSRMatrixInternal<IdType, IdType> *block_csr_array, const DType *B,
    const DType *E, DType *C, IdType *argB, IdType *argE, bool has_idx,
    IdType N, IdType num_M_blocks, IdType num_K_blocks, IdType M_block_size,
    libxsmm_meltwfunction_opreduce_vecs_idx kernel) {
  const DType *in_matrix1 = B;
  const DType *in_matrix2 = E;
  DType *output = C;
  IdType *out_matrix1 = argB;
  IdType *out_matrix2 = argE;

#pragma omp parallel
  {
    for (IdType k = 0; k < num_K_blocks; k++) {
#pragma omp for schedule(dynamic)
      for (IdType m = 0; m < num_M_blocks; m++) {
        CSRMatrixInternal<IdType, IdType> cur_csr =
            block_csr_array[m * num_K_blocks + k];

        const IdType M_start = m * M_block_size;
        for (IdType i = 0; i < cur_csr.num_rows; i++) {
          const IdType row_start = cur_csr.indptr[i];
          const IdType row_end = cur_csr.indptr[i + 1];
          const IdType dst = i + M_start;

          libxsmm_meltw_opreduce_vecs_idx_param params;
          params.n = row_end - row_start;
          params.indices = &cur_csr.indices[row_start];
          params.in_matrix = in_matrix1;
          params.out_vec = &output[dst * N];
          params.argop_off_vec_0 = &out_matrix1[dst * N];
          params.argop_off_vec_1 = &out_matrix2[dst * N];
          params.scale_vals = nullptr;
          if (has_idx) {
            params.in_matrix2 = in_matrix2;
            params.indices2 = &cur_csr.data[row_start];
          } else {
            params.in_matrix2 = &in_matrix2[row_start * N];
          }
          kernel(&params);
        }
      }
    }
  }
}

/**
 * @brief Free the tiled CSR matrix data.
 * @param block_csr_array The array containing csr matrices of all blocks.
 * @param num_M_blocks Number of blocks to create along the rows of adjacency
 *        matrix.
 * @param num_K_blocks Number of blocks to create along the columns of adjacency
 *        matrix.
 * @param use_lhs Whether to use lhs.
 * @param use_rhs Whether to use rhs.
 */
template <typename IdType>
inline void SpMMFreeBlocks(
    CSRMatrixInternal<IdType, IdType> *block_csr_array, IdType num_M_blocks,
    IdType num_K_blocks, bool use_lhs, bool use_rhs) {
  if (num_K_blocks > 1) {
    free(block_csr_array[0].indptr);
    if (use_lhs) free(block_csr_array[0].indices);
    if (use_rhs) free(block_csr_array[0].data);
  }
  free(block_csr_array);
}

/**
 * @brief Optimized CPU kernel of SpMM-Sum/Max/Min on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes.
 * @param arge Arg-Min/Max on edges.
 * @note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op, typename Redop>
void SpMMRedopCsrOpt(
    const BcastOff &bcast, const CSRMatrix &csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  int32_t llc_size = GetLLCSize();

#ifdef DEBUG
  uint64_t startTick, endTick;
  startTick = __rdtsc();
#endif  // DEBUG

  const bool has_idx = !IsNullArray(csr.data);

  DType *C = out.Ptr<DType>();
  const DType *B = ufeat.Ptr<DType>();
  const DType *E = efeat.Ptr<DType>();
  IdType *argB, *argE;
  if (std::is_same<Redop, op::Max<DType>>::value ||
      std::is_same<Redop, op::Min<DType>>::value) {
    argB = argu.Ptr<IdType>();
    argE = arge.Ptr<IdType>();
  }

  const int nthreads = omp_get_max_threads();
  const IdType M = csr.num_rows;
  const IdType N = bcast.out_len;
  const IdType K = csr.num_cols;
  const IdType *indptr = csr.indptr.Ptr<IdType>();
  CHECK_NOTNULL(indptr);
  const IdType total_nnz = indptr[M];
  if (M <= 0 || K <= 0 || N <= 0 || total_nnz <= 0) return;

  const double avg_degree = total_nnz * 1.0 / M;
  const double nnz_prob = avg_degree / K;

  IdType K_block_size = std::min(
      (int64_t)K,
      (int64_t)(llc_size / (N * sizeof(DType) * nnz_prob * BLOCKING_HEURISTIC_PARAM)));  // NOLINT
  IdType M_block_size = M / (nthreads * NUM_BLOCKS_PER_THREAD);
  if (M_block_size == 0) M_block_size = 1;
  if (K_block_size == 0) K_block_size = 1;

  IdType num_M_blocks = (M + M_block_size - 1) / M_block_size;
  IdType num_K_blocks = (K + K_block_size - 1) / K_block_size;

  CSRMatrixInternal<IdType, IdType> *block_csr_array =
      (CSRMatrixInternal<IdType, IdType> *)aligned_alloc(
          64, sizeof(CSRMatrixInternal<IdType, IdType>) * num_M_blocks *
                  num_K_blocks);

#ifdef DEBUG
  endTick = __rdtsc();
  if (std::is_same<Redop, op::Max<DType>>::value) {
    LOG(INFO) << "Redop = Max";
  } else if (std::is_same<Redop, op::Min<DType>>::value) {
    LOG(INFO) << "Redop = Min";
  } else if (std::is_same<Redop, op::Add<DType>>::value) {
    LOG(INFO) << "Redop = Add";
  }
  LOG(INFO) << "nthreads = " << nthreads << ", llc_size = " << llc_size;
  LOG(INFO) << "M = " << M << ", K = " << K << ", N = " << N;
  LOG(INFO) << "use_lhs = " << Op::use_lhs << ", use_rhs = " << Op::use_rhs;
  LOG(INFO) << "total_nnz = " << total_nnz << ", avg_degree = " << avg_degree;
  LOG(INFO) << "has_idx = " << has_idx;
  LOG(INFO) << "nnz_prob = " << nnz_prob;
  LOG(INFO) << "K_block_size = " << K_block_size
            << ", M_block_size = " << M_block_size;
  LOG(INFO) << "num_K_blocks = " << num_K_blocks
            << ", num_M_blocks = " << num_M_blocks;
  LOG(INFO) << "stage0 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif  // DEBUG

  SpMMCreateBlocks(
      csr, block_csr_array, num_M_blocks, num_K_blocks, M_block_size,
      K_block_size, Op::use_lhs, Op::use_rhs);

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage1 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif  // DEBUG

  libxsmm_meltwfunction_opreduce_vecs_idx kernel = nullptr;
  if (std::is_same<Redop, op::Max<DType>>::value) {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(
        has_idx, N, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX, true);
  } else if (std::is_same<Redop, op::Min<DType>>::value) {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(
        has_idx, N, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN, true);
  } else if (std::is_same<Redop, op::Add<DType>>::value) {
    kernel = SpMMCreateLibxsmmKernel<IdType, DType, Op>(
        has_idx, N, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM, false);
  }

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage2 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif  // DEBUG

  if (std::is_same<Redop, op::Max<DType>>::value ||
      std::is_same<Redop, op::Min<DType>>::value) {
    SpMMBlockwiseOpCmp<IdType, DType, Op, Redop>(
        block_csr_array, B, E, C, argB, argE, has_idx, N, num_M_blocks,
        num_K_blocks, M_block_size, kernel);
  } else {
    SpMMBlockwiseOpSum(
        block_csr_array, B, E, C, has_idx, N, num_M_blocks, num_K_blocks,
        M_block_size, kernel);
  }

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage3 ticks = " << (endTick - startTick);
  startTick = __rdtsc();
#endif  // DEBUG

  SpMMFreeBlocks(
      block_csr_array, num_M_blocks, num_K_blocks, Op::use_lhs, Op::use_rhs);

#ifdef DEBUG
  endTick = __rdtsc();
  LOG(INFO) << "stage4 ticks = " << (endTick - startTick);
#endif  // DEBUG
}

/**
 * @brief Optimized CPU kernel of SpMM-Sum on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCsrLibxsmm(
    const BcastOff &bcast, const CSRMatrix &csr, NDArray ufeat, NDArray efeat,
    NDArray out) {
  NDArray dummy;
  SpMMRedopCsrOpt<IdType, DType, Op, op::Add<DType>>(
      bcast, csr, ufeat, efeat, out, dummy, dummy);
}

/**
 * @brief Optimized CPU kernel of SpMM-Min/Max on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes.
 * @param arge Arg-Min/Max on edges.
 * @note it uses libxsmm, blocking and dynamic thread scheduling.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsrLibxsmm(
    const BcastOff &bcast, const CSRMatrix &csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  SpMMRedopCsrOpt<IdType, DType, Op, Cmp>(
      bcast, csr, ufeat, efeat, out, argu, arge);
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // USE_LIBXSMM
#endif  // _WIN32

#endif  // DGL_ARRAY_CPU_SPMM_BLOCKING_LIBXSMM_H_
