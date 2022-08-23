/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/context.cc
 * \brief DGL aten context
 */

#include "./context.h"

namespace dgl {
namespace aten {

Context::Context() = default;

Context& globalContext() {
    static Context globalContext_;
    return globalContext_;
}

void Context::setLibxsmm(bool b) {
    _libxsmm = b;
}

bool Context::libxsmm() const {
    return _libxsmm;
}

}  // namespace aten
}  // namespace dgl
