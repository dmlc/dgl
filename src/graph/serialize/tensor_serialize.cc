/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/tensor_serialize.cc
 * @brief Graph serialization implementation
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

constexpr uint64_t kDGLSerialize_Tensors = 0xDD5A9FBE3FA2443F;

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_SaveNDArrayDict")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      auto fs = std::unique_ptr<dmlc::Stream>(
          dmlc::Stream::Create(filename.c_str(), "w"));
      CHECK(fs) << "Filename is invalid";
      fs->Write(kDGLSerialize_Tensors);
      bool empty_dict = args[2];
      Map<std::string, Value> nd_dict;
      if (!empty_dict) {
        nd_dict = args[1];
      }
      std::vector<NamedTensor> namedTensors;
      fs->Write(static_cast<uint64_t>(nd_dict.size()));
      for (auto kv : nd_dict) {
        NDArray ndarray = static_cast<NDArray>(kv.second->data);
        namedTensors.emplace_back(kv.first, ndarray);
      }
      fs->Write(namedTensors);
      *rv = true;
    });

DGL_REGISTER_GLOBAL("data.tensor_serialize._CAPI_LoadNDArrayDict")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      auto fs = std::unique_ptr<dmlc::Stream>(
          dmlc::Stream::Create(filename.c_str(), "r"));
      CHECK(fs) << "Filename is invalid or file doesn't exists";
      uint64_t magincNum, num_elements;
      CHECK(fs->Read(&magincNum)) << "Invalid file";
      CHECK_EQ(magincNum, kDGLSerialize_Tensors) << "Invalid DGL tensor file";
      CHECK(fs->Read(&num_elements)) << "Invalid num of elements";
      Map<std::string, Value> nd_dict;
      std::vector<NamedTensor> namedTensors;
      fs->Read(&namedTensors);
      for (auto kv : namedTensors) {
        Value ndarray = Value(MakeValue(kv.second));
        nd_dict.Set(kv.first, ndarray);
      }
      *rv = nd_dict;
    });

}  // namespace serialize
}  // namespace dgl
