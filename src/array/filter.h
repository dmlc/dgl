/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/filter.h
 * \brief Object for selecting items in a set, or selecting items not in a set.
 */


#ifndef DGL_ARRAY_FILTER_H_
#define DGL_ARRAY_FILTER_H_

#include <dgl/runtime/object.h>
#include <dgl/array.h>

namespace dgl {
namespace array {

class Filter : public runtime::Object {
 public:
  static constexpr const char* _type_key = "array.Filter";
  DGL_DECLARE_OBJECT_TYPE_INFO(Filter, Object);

  /**
   * @brief From the test set of items, get those which are included by this
   * filter.
   *
   * @param test The set of items to check for.
   *
   * @return The subset of items from `test` that are selected by this filter.
   */
  virtual IdArray include(
      IdArray test) = 0;

  /**
   * @brief From the test set of items, get those which are excluded by this
   * filter.
   *
   * @param test The set of items to check for.
   *
   * @return The subset of items from `test` that are not selected by this
   * filter.
   */
  virtual IdArray exclude(
      IdArray test) = 0;
};

DGL_DEFINE_OBJECT_REF(FilterRef, Filter);

}  // namespace array
}  // namespace dgl

#endif // DGL_ARRAY_FILTER_H_

