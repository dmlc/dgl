/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/utils.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_UTILS_H_
#define DGL_GRAPH_SERIALIZE_UTILS_H_

#include <dgl/array.h>
#include <dgl/graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../../c_api_common.h"

using dgl::runtime::NDArray;
using namespace dgl::runtime;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_UTILS_H_
