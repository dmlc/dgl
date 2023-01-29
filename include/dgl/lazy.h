/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/lazy.h
 * @brief Lazy object that will be materialized only when being queried.
 */
#ifndef DGL_LAZY_H_
#define DGL_LAZY_H_

#include <memory>

namespace dgl {

/**
 * @brief Lazy object that will be materialized only when being queried.
 *
 * The object should be immutable -- no mutation once materialized.
 * The object is currently not threaad safe.
 */
template <typename T>
class Lazy {
 public:
  /** @brief default constructor to construct a lazy object */
  Lazy() {}

  /**
   * @brief constructor to construct an object with given value (non-lazy case)
   */
  explicit Lazy(const T& val) : ptr_(new T(val)) {}

  /** @brief destructor */
  ~Lazy() = default;

  /**
   * @brief Get the value of this object. If the object has not been
   *        instantiated, using the provided function to create it.
   * @param fn The creator function.
   * @return the object value.
   */
  template <typename Fn>
  const T& Get(Fn fn) {
    if (!ptr_) {
      ptr_.reset(new T(fn()));
    }
    return *ptr_;
  }

 private:
  /** @brief the internal data pointer */
  std::shared_ptr<T> ptr_{nullptr};
};

}  // namespace dgl

#endif  // DGL_LAZY_H_
