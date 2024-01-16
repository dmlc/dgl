/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/max_uva_threads.cc
 * @brief Max uva threads variable setter function.
 */
#include "./max_uva_threads.h"

namespace graphbolt {
namespace cuda {

void set_max_uva_threads(int64_t count) { max_uva_threads = count; }

}  // namespace cuda
}  // namespace graphbolt
