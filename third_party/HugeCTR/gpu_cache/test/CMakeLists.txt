#
# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)
file(GLOB gpu_cache_test_src
  cache_op_sol_test.cu
  ../../HugeCTR/src/hps/embedding_cache_gpu.cu
)

add_executable(cache_op_sol_test ${gpu_cache_test_src})
target_compile_features(cache_op_sol_test PUBLIC cxx_std_17)
target_link_libraries(cache_op_sol_test PUBLIC gpu_cache)
target_link_libraries(cache_op_sol_test PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(cache_op_sol_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cache_op_sol_test PROPERTIES CUDA_ARCHITECTURES OFF)

