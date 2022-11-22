/**
 *  Copyright (c) 2022 by Contributors
 * @file dgl_headers.h
 * @brief DGL headers used in the sparse library. This is a workaround to
 * avoid the macro naming conflict between dmlc/logging.h and torch logger. This
 * file includes all the DGL headers used in the sparse library and
 * undefines logging macros defined in dmlc/logging.h. There are two rules to
 * use this file. (1) All DGL headers used in the sparse library should be and
 * only be registered in this file. (2) When including Pytorch headers, this
 * file should be included in advance.
 */
#ifndef SPARSE_DGL_HEADERS_H_
#define SPARSE_DGL_HEADERS_H_

#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/dlpack_convert.h>
#include <dmlc/logging.h>

#undef CHECK
#undef CHECK_OP
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#undef CHECK_NOTNULL
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT
#undef DCHECK_NOTNULL
#undef VLOG
#undef LOG
#undef DLOG
#undef LOG_IF

// For Pytorch version later than 1.12, redefine CHECK_* to TORCH_CHECK_*.
#if !(TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 12)
#define CHECK_EQ(val1, val2) TORCH_CHECK_EQ(val1, val2)
#define CHECK_NE(val1, val2) TORCH_CHECK_NE(val1, val2)
#define CHECK_LE(val1, val2) TORCH_CHECK_LE(val1, val2)
#define CHECK_LT(val1, val2) TORCH_CHECK_LT(val1, val2)
#define CHECK_GE(val1, val2) TORCH_CHECK_GE(val1, val2)
#define CHECK_GT(val1, val2) TORCH_CHECK_GT(val1, val2)
#endif

#endif  // SPARSE_DGL_HEADERS_H_
