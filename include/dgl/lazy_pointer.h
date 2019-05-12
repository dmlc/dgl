/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/lazy_pointer.h
 * \brief Lazy object pointer.
 */
#ifndef DGL_LAZY_POINTER_H_
#define DGL_LAZY_POINTER_H_

#include <functional>
#include <memory>

namespace dgl {

/*
 * !\brief 
 *
 * Note: the class is not thread-safe.
 */
template <typename T>
class LazyPointer {
 public:
  /*! \brief default constructor */
  LazyPointer() {}

  /*! \brief default destructor */
  ~LazyPointer() = default;

  /*! \brief default copy constructor */
  LazyPointer(const LazyPointer<T>& other) = default;

  /*! \brief default assign constructor */
  LazyPointer& operator=(const LazyPointer<T>& other) = default;

  /*! \brief Get the pointer and alloc if not yet. */
  template <typename FCreate>
  std::shared_ptr<T> Get(FCreate creator) {
    if (!ptr_) {
      ptr_.reset(creator());
    }
    return ptr_;
  }

  /*! \brief reset the internal pointer and release memory */
  void Reset() {
    ptr_ = nullptr;
  }

  /*! \brief directly return the pointer */
  std::shared_ptr<T> operator->() const {
    return ptr_;
  }

  /*! \brief conversion operator */
  operator bool() const {
    return ptr_;
  }

  /*! \brief equal operator */
  bool operator==(const LazyPointer<T>& other) const {
    return ptr_ == other.ptr_;
  }

  /*! \brief not equal operator */
  bool operator!=(const LazyPointer<T>& other) const {
    return ptr_ != other.ptr_;
  }

 private:
  // internal pointer
  std::shared_ptr<T> ptr_{nullptr};
};

}  // namespace dgl

#endif  // DGL_LAZY_POINTER_H_
