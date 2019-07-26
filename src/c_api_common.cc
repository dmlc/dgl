/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_runtime_api.cc
 * \brief DGL C API common implementations
 */
#include "c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

PackedFunc ConvertNDArrayVectorToPackedFunc(const std::vector<NDArray>& vec) {
    auto body = [vec](DGLArgs args, DGLRetValue* rv) {
        const uint64_t which = args[0];
        if (which >= vec.size()) {
            LOG(FATAL) << "invalid choice";
        } else {
            *rv = std::move(vec[which]);
        }
    };
    return PackedFunc(body);
}

}  // namespace dgl
