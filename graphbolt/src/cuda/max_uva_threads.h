/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
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
