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
 * @file io_uring.h
 * @brief io_uring related functions.
 */
#ifndef GRAPHBOLT_IO_URING_H_
#define GRAPHBOLT_IO_URING_H_

#include <cstdint>
#include <optional>

namespace graphbolt {
namespace io_uring {

bool IsAvailable();

/** @brief Set a limit on # background io_uring threads. */
inline std::optional<int64_t> num_threads;

/**
 * @brief Set the number of background io_uring threads.
 */
void SetNumThreads(int64_t count);

}  // namespace io_uring
}  // namespace graphbolt

#endif  // GRAPHBOLT_IO_URING_H_
