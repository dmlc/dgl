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

#if defined(WIN32) || defined(_WIN32)
#define TA_EXPORTS __declspec(dllexport)
#else
#define TA_EXPORTS
#endif

#endif  // TENSORADAPTER_H_
