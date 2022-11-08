/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/macro.cuh
 * @brief Macro to call SPMM/SDDMM cuda kernels.
 */
#ifndef DGL_ARRAY_CUDA_MACRO_CUH_
#define DGL_ARRAY_CUDA_MACRO_CUH_

///////////////////////// Dispatchers //////////////////////////

/* Macro used for switching between broadcasting and non-broadcasting kernels.
 * It also copies the auxiliary information for calculating broadcasting offsets
 * to GPU.
 */
#define BCAST_IDX_CTX_SWITCH(BCAST, EDGE_MAP, CTX, LHS_OFF, RHS_OFF, ...)     \
  do {                                                                        \
    const BcastOff &info = (BCAST);                                           \
    if (!info.use_bcast) {                                                    \
      constexpr bool UseBcast = false;                                        \
      if ((EDGE_MAP)) {                                                       \
        constexpr bool UseIdx = true;                                         \
        { __VA_ARGS__ }                                                       \
      } else {                                                                \
        constexpr bool UseIdx = false;                                        \
        { __VA_ARGS__ }                                                       \
      }                                                                       \
    } else {                                                                  \
      constexpr bool UseBcast = true;                                         \
      const DGLContext ctx = (CTX);                                           \
      const auto device = runtime::DeviceAPI::Get(ctx);                       \
      (LHS_OFF) = static_cast<int64_t *>(device->AllocWorkspace(              \
          ctx, sizeof(int64_t) * info.lhs_offset.size()));                    \
      CUDA_CALL(cudaMemcpy(                                                   \
          (LHS_OFF), &info.lhs_offset[0],                                     \
          sizeof(int64_t) * info.lhs_offset.size(), cudaMemcpyHostToDevice)); \
      (RHS_OFF) = static_cast<int64_t *>(device->AllocWorkspace(              \
          ctx, sizeof(int64_t) * info.rhs_offset.size()));                    \
      CUDA_CALL(cudaMemcpy(                                                   \
          (RHS_OFF), &info.rhs_offset[0],                                     \
          sizeof(int64_t) * info.rhs_offset.size(), cudaMemcpyHostToDevice)); \
      if ((EDGE_MAP)) {                                                       \
        constexpr bool UseIdx = true;                                         \
        { __VA_ARGS__ }                                                       \
      } else {                                                                \
        constexpr bool UseIdx = false;                                        \
        { __VA_ARGS__ }                                                       \
      }                                                                       \
      device->FreeWorkspace(ctx, (LHS_OFF));                                  \
      device->FreeWorkspace(ctx, (RHS_OFF));                                  \
    }                                                                         \
  } while (0)

#endif  // DGL_ARRAY_CUDA_MACRO_CUH_
