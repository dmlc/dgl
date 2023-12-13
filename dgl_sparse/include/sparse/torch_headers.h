/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/torch_headers.h
 * @brief Pytorch headers used in the sparse library. Since Pytorch 2.1.0
 * introduced a dependency on <windows.h>, we need to define NOMINMAX to avoid
 * the conflict with std::min/std::max macros before including Pytorch headers.
 * See more in
 * https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c.
 */

#ifndef SPARSE_TORCH_HEADERS_H_
#define SPARSE_TORCH_HEADERS_H_

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#endif  // _WIN32

#include <ATen/DLConvertor.h>
#include <c10/util/Logging.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#ifdef _WIN32
#undef NOMINMAX
#endif  // _WIN32

#endif  // SPARSE_TORCH_HEADERS_H_
