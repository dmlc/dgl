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

#ifdef WIN32
#define TA_EXPORTS __declspec(dllexport)
#else
#define TA_EXPORTS
#endif

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
TA_EXPORTS DLManagedTensor* TAempty(
    std::vector<int64_t> shape, DLDataType dtype, DLContext ctx);

/*!
 * \brief The macro that calls an entrypoint with the signature of the given function.
 *
 * Use it like:
 *
 * <code>
 *   auto handle = dlopen("tensoradapter_torch.so");
 *   auto entry = dlsym(handle, "TAempty");
 *   auto result = TA_DISPATCH(tensoradapter::TAempty, entry, shape, dtype, ctx);
 * </code>
 */
#define TA_DISPATCH(func, entry, ...) \
  ((*reinterpret_cast<decltype(&func)>(entry))(__VA_ARGS__))

}

};  // namespace tensoradapter

#endif  // TENSORADAPTER_H_
