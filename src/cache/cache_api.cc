/*!
 *  Copyright (c) 2018 by Contributors
 * \file cache/cache.cc
 * \brief DGL cache implementation
 */

#include <dgl/graph.h>
#include <dgl/cache.h>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

namespace dgl {

DGL_REGISTER_GLOBAL("frame_cache._CAPI_DGLCacheLookup")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray cached_ids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    std::vector<IdArray> arrs;
    arrs.push_back(IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1])));
    *rv = ConvertNDArrayVectorToPackedFunc(cache::SearchCachedIds(cached_ids, arrs));
  });

#define DEF_CACHE_LOOKUP(num_inputs) DGL_REGISTER_GLOBAL(                               \
    std::string("frame_cache._CAPI_DGLCacheLookup") + std::to_string(num_inputs))       \
    .set_body([] (DGLArgs args, DGLRetValue* rv) {                                      \
      const IdArray cached_ids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));\
      std::vector<IdArray> arrs;                                                        \
      for (int i = 0; i < num_inputs; i++) {                                            \
        arrs.push_back(IdArray::FromDLPack(CreateTmpDLManagedTensor(args[i + 1])));     \
      }                                                                                 \
      *rv = ConvertNDArrayVectorToPackedFunc(cache::SearchCachedIds(cached_ids, arrs)); \
    })

DEF_CACHE_LOOKUP(2);
DEF_CACHE_LOOKUP(4);
DEF_CACHE_LOOKUP(8);
DEF_CACHE_LOOKUP(16);
DEF_CACHE_LOOKUP(32);

}  // namespace dgl
