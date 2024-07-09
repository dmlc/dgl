/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file detect_io_uring.cc
 * @brief Check whether io_uring is available on the system.
 */
#ifdef HAVE_LIBRARY_LIBURING
#include <errno.h>
#include <linux/io_uring.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace graphbolt {
namespace io_uring {

bool IsAvailable() {
#ifdef HAVE_LIBRARY_LIBURING
  if (cached_is_available.has_value()) return cached_is_available.value();
  // https://unix.stackexchange.com/a/596284/314554
  cached_is_available = !(
      syscall(__NR_io_uring_register, 0, IORING_UNREGISTER_BUFFERS, NULL, 0) &&
      errno == ENOSYS);
  return cached_is_available.value();
#else
  return false;
#endif
}

}  // namespace io_uring
}  // namespace graphbolt
