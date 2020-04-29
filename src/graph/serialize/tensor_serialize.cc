/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/tensor_serialize.cc
 * \brief Graph serialization implementation
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>

#include "../../c_api_common.h"

using namespace dgl::runtime;
using dmlc::SeekStream;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_SaveNDArrayDict")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    Map<std::string, Value> nd_dict = args[1];
    std::vector<NamedTensor> namedTensors;
    for (auto kv : nd_dict) {
      NDArray ndarray = static_cast<NDArray>(kv.second->data);
      namedTensors.emplace_back(kv.first, ndarray);
    }
    auto *fs = dmlc::Stream::Create(filename.c_str(), "w");
    CHECK(fs) << "Filename is invalid";
    fs->Write(namedTensors);
    delete fs;
    *rv = true;
  });

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_LoadNDArrayDict")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    Map<std::string, Value> nd_dict;
    std::vector<NamedTensor> namedTensors;
    auto *fs = dmlc::Stream::Create(filename.c_str(), "r");
    CHECK(fs) << "Filename is invalid or file doesn't exists";
    fs->Read(&namedTensors);
    for (auto kv : namedTensors) {
      Value ndarray = Value(MakeValue(kv.second));
      nd_dict.Set(kv.first, ndarray);
    }
    delete fs;
    *rv = nd_dict;
  });

}  // namespace serialize
}  // namespace dgl
