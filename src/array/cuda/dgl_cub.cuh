/*!
 *  Copyright (c) 2021 by Contributors
 * \file cuda_common.h
 * \brief Wrapper to place cub in dgl namespace. 
 */

#ifndef DGL_ARRAY_CUDA_DGL_CUB_CUH_
#define DGL_ARRAY_CUDA_DGL_CUB_CUH_

// include cub in a safe manner
#define CUB_NS_PREFIX namespace dgl {
#define CUB_NS_POSTFIX }
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX

#endif
