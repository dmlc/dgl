/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/config.cc
 * \brief DGL runtime config
 */

#include <dgl/runtime/config.h>

namespace dgl {
namespace runtime {

void Config::enableLibxsmm(bool b) {
    _libxsmm = b;
}

bool Config::isLibxsmmAvailable() const {
    return _libxsmm;
}

}  // namespace runtime
}  // namespace dgl
