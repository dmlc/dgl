/*!
 *  Copyright (c) 2020 by Contributors
 * \file tensoradapter.h
 * \brief Header file for functions exposed by the adapter library.
 *
 * Functions in this library must be exported with extern "C" so that DGL can locate
 * them with dlsym(3) (or GetProcAddress on Windows).
 */

#ifndef TENSORADAPTER_H_
#define TENSORADAPTER_H_

#include <dlpack/dlpack.h>
#include <vector>

namespace tensoradapter {

extern "C" {

/*!
 * \brief Allocate an empty tensor
 *
 * \param shape The shape
 * \param dtype The data type
 * \param ctx The device
 * \return The allocated tensor
 */
DLManagedTensor* TAempty(
    std::vector<int64_t> shape, DLDataType dtype, DLContext ctx);

}

};  // namespace tensoradapter

#endif  // TENSORADAPTER_H_
