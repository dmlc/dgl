/**
 *  Copyright (c) 2020 by Contributors
 * @file tensoradapter_exports.h
 * @brief Header file for functions exposed by the adapter library.
 */

#ifndef TENSORADAPTER_EXPORTS_H_
#define TENSORADAPTER_EXPORTS_H_

#if defined(WIN32) || defined(_WIN32)
#define TA_EXPORTS __declspec(dllexport)
#else
#define TA_EXPORTS
#endif

#endif  // TENSORADAPTER_EXPORTS_H_
