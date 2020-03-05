/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/lazy.h
 * \brief Lazy object that will be materialized only when being queried.
 */
#ifndef DGL_LAZY_H_
#define DGL_LAZY_H_

#include <memory>
#include <mutex>
#include <atomic>

namespace dgl {

/*!
 * \brief Lazy initialization
 */
template <typename T, typename Fn>
const std::shared_ptr<T> LazyInit(std::shared_ptr<T>* ptr, std::mutex *mutex, Fn func) {
  std::shared_ptr<T> result;

  while (!(result = std::atomic_load_explicit(ptr, std::memory_order_acquire))) {
    std::lock_guard<std::mutex> lock(*mutex);
    result = *ptr;

    if (!result) {
      result.reset(new T(func()));
      std::atomic_store_explicit(ptr, result, std::memory_order_release);
    }
  }

  return result;
}

/*!
 * \brief Lazy object that will be materialized only when being queried.
 *
 * The object should be immutable -- no mutation once materialized.
 *
 * \note Implementation taken from
 * https://www.justsoftwaresolutions.co.uk/threading/multithreading-in-c++0x-part-6-double-checked-locking.html
 */
template <typename T>
class Lazy {
 public:
  /*!\brief default constructor to construct a lazy object */
  Lazy() {}

  /*!\brief constructor to construct an object with given value (non-lazy case) */
  explicit Lazy(const T& val): ptr_(new T(val)), preinitialized_(true) {}

  /*!\brief copy constructor which ignores mutex */
  Lazy(const Lazy& other) {
    ptr_ = other.ptr_;
    preinitialized_ = other.preinitialized_;
  }

  /*!\brief assignment operator which ignores mutex */
  Lazy& operator=(const Lazy other) {
    this->ptr_ = other.ptr_;
    this->preinitialized_ = other.preinitialized_;
    return *this;
  }

  /*!\brief destructor */
  ~Lazy() = default;

  /*!
   * \brief Get the value of this object. If the object has not been instantiated,
   *        using the provided function to create it.
   * \param fn The creator function.
   * \return the object value.
   */
  template <typename Fn>
  const T& Get(Fn fn) {
    return *LazyInit(&ptr_, &mutex_, fn);
  }

  /*!
   * \brief Determine whether the object is initialized during construction.
   */
  bool PreInitialized() const {
    return preinitialized_;
  }

  /*!
   * \brief Get the value of this object.  Assumes that the object is initialized during
   * construction.
   */
  const T& Get() const {
    CHECK(preinitialized_) << "not preinitialized";
    return *ptr_;
  }

 private:
  /*!\brief the internal data pointer */
  std::shared_ptr<T> ptr_{nullptr};
  /*!\brief mutex */
  mutable std::mutex mutex_;
  /*!\brief whether the lazy object is pre-initialized during construction */
  bool preinitialized_;
};

}  // namespace dgl

#endif  // DGL_LAZY_H_
