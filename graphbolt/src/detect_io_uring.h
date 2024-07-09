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
 * @file detect_io_uring.h
 * @brief Check whether io_uring is available on the system.
 */
#ifndef GRAPHBOLT_DETECT_IO_URING_H_
#define GRAPHBOLT_DETECT_IO_URING_H_

namespace graphbolt {
namespace io_uring {

/** @brief The cached value of whether io_uring is available. */
inline std::optional<bool> cached_is_available;

bool IsAvailable();

}  // namespace io_uring
}  // namespace graphbolt

#endif  // GRAPHBOLT_DETECT_IO_URING_H_
