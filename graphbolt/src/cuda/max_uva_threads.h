/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/max_uva_threads.h
 * @brief Max uva threads variable declaration.
 */
#ifndef GRAPHBOLT_MAX_UVA_THREADS_H_
#define GRAPHBOLT_MAX_UVA_THREADS_H_

#include <cstdint>
#include <optional>

namespace graphbolt {
namespace cuda {

/** @brief Set a limit on the number of CUDA threads for UVA accesses. */
inline std::optional<int64_t> max_uva_threads;

void set_max_uva_threads(int64_t count);

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_MAX_UVA_THREADS_H_
