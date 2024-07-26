/**
 *   Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
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
 * @file utils.cc
 * @brief Graphbolt utils implementations.
 */
#include "./utils.h"

#include <optional>

namespace graphbolt {
namespace utils {

namespace {
std::optional<int64_t> worker_id;
}

std::optional<int64_t> GetWorkerId() { return worker_id; }

void SetWorkerId(int64_t worker_id_value) { worker_id = worker_id_value; }

}  // namespace utils
}  // namespace graphbolt
