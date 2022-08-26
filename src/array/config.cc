/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/config.cc
 * \brief DGL aten config
 */

#include "./config.h"

namespace dgl {
namespace aten {

void Config::enableLibxsmm(bool b) {
    _libxsmm = b;
}

bool Config::isLibxsmmAvailable() const {
    return _libxsmm;
}

}  // namespace aten
}  // namespace dgl
