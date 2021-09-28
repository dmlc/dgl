/*!
 *  Copyright 2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * \file graph/transform/cuda_compact_graph.cu
 * \brief Functions to find and eliminate the common isolated nodes across
 * all given graphs with the same set of nodes.
 */


#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>
#include <cuda_runtime.h>
#include <utility>
#include <algorithm>
#include <memory>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../../runtime/cuda/cuda_hashtable.cuh"
#include "../../heterograph.h"
#include "../compact.h"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;

namespace dgl {
namespace transform {

namespace {
  // implementation here


}  // namespace

template<>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDLGPU, int32_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int32_t>(graphs, always_preserve);
}

template<>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDLGPU, int64_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int64_t>(graphs, always_preserve);
}

}  // namespace transform
}  // namespace dgl
