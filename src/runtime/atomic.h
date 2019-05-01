/*!
 *  Copyright (c) 2019 by Contributors
 * \file c_atomic.h
 * \brief DGL atomic operations
 */
#ifndef DGL_ATOMIC_H_
#define DGL_ATOMIC_H_

#include <stdatomic.h>
#include <atomic>

namespace dgl {

template<class T>
T atomic_read(volatile T *val) {
  T ret_val;
  __atomic_load(val, &ret_val, memory_order_seq_cst);
  return ret_val;
}

class ReadWriteLock {
  volatile int &lock;
  int tmp_val;

  static bool is_write_mode(int lock_val) {
    return lock_val < 0;
  }

  void enter_write_mode() {
    int prev;
    do {
      prev = __atomic_fetch_or(&lock, 1<<31, memory_order_seq_cst);
      // If the previous value already has write lock set up,
      // we have to wait until the write lock is reset.
    } while(prev < 0);
  }

  void leave_write_mode() {
    int prev = __atomic_fetch_and(&lock, ~(1<<31), memory_order_seq_cst);
    // when leaving the write mode, we have to make sure the write lock was
    // set up.
    CHECK(prev < 0);
  }
public:
  ReadWriteLock(int &_lock): lock(_lock) {
  }

  void ReadLock() {
    tmp_val = atomic_read(&lock);
  }

  bool ReadUnlock() {
    int tmp_val2 = atomic_read(&lock);
    // If the first read and the second read has different values,
    // it means the row has been modified during reading. We should read it again.
    // Even if they are the same, we still need to check if the lock is in the write mode,
    // which means another process is locking the row for modification.
    return tmp_val == tmp_val2 && !is_write_mode(tmp_val);
  }

  void WriteLock() {
    enter_write_mode();
  }

  void WriteUnlock() {
    __atomic_fetch_add(&lock, 1, memory_order_seq_cst);
    leave_write_mode();
  }
};

}  // namespace dgl

#endif  // DGL_ATOMIC_H_
