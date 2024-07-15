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
 * @file graphbolt/async.h
 * @brief Provides asynchronous task utilities for GraphBolt.
 */
#ifndef GRAPHBOLT_ASYNC_H_
#define GRAPHBOLT_ASYNC_H_

#include <ATen/Parallel.h>
#include <torch/script.h>

#include <future>

namespace graphbolt {

template <typename T>
class Future : public torch::CustomClassHolder {
 public:
  Future(std::future<T>&& future) : future_(std::move(future)) {}

  Future() = default;

  T Wait() { return future_.get(); }

 private:
  std::future<T> future_;
};

template <typename F>
auto async(F function) {
  using T = decltype(function());
  auto promise = std::make_shared<std::promise<T>>();
  auto future = promise->get_future();
  at::launch([=]() { promise->set_value(function()); });
  return c10::make_intrusive<Future<T>>(std::move(future));
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_ASYNC_H_
