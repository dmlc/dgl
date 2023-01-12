/**
 *  Copyright (c) 2017 by Contributors
 * @file meta_data.h
 * @brief Meta data related utilities
 */
#ifndef DGL_RUNTIME_META_DATA_H_
#define DGL_RUNTIME_META_DATA_H_

#include <dgl/runtime/packed_func.h>
#include <dmlc/io.h>
#include <dmlc/json.h>

#include <string>
#include <vector>

#include "runtime_base.h"

namespace dgl {
namespace runtime {

/** @brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<DGLDataType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(dmlc::JSONWriter *writer) const;
  void Load(dmlc::JSONReader *reader);
  void Save(dmlc::Stream *writer) const;
  bool Load(dmlc::Stream *reader);
};
}  // namespace runtime
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::dgl::runtime::FunctionInfo, true);
}  // namespace dmlc
#endif  // DGL_RUNTIME_META_DATA_H_
