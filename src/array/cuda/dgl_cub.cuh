/*!
 *  Copyright (c) 2021 by Contributors
 * \file cuda_common.h
 * \brief Wrapper to place cub in dgl namespace. 
 */

#ifndef DGL_ARRAY_CUDA_DGL_CUB_CUH_
#define DGL_ARRAY_CUDA_DGL_CUB_CUH_

// This should be defined in CMakeLists.txt
#ifndef THRUST_CUB_WRAPPED_NAMESPACE
static_assert(false, "THRUST_CUB_WRAPPED_NAMESPACE must be defined for DGL.");
#endif

#include "cub/cub.cuh"

static_assert(CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14,
    "CUB version is too old. Please update to 1.14 or later.");

#endif
