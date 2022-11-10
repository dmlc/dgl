/**
 *  Copyright (c) 2019 by Contributors
 * @file common.h
 * @brief This file provide basic facilities for string
 * to make programming convenient.
 */
#ifndef DGL_RPC_NETWORK_COMMON_H_
#define DGL_RPC_NETWORK_COMMON_H_

#include <dmlc/logging.h>

#include <set>
#include <string>
#include <vector>

namespace dgl {
namespace network {

//------------------------------------------------------------------------------
// Subdivide string |full| into substrings according to delimitors
// given in |delim|.  |delim| should pointing to a string including
// one or more characters.  Each character is considerred a possible
// delimitor. For example:
//
//   vector<string> substrings;
//   SplitStringUsing("apple orange\tbanana", "\t ", &substrings);
//
// results in three substrings:
//
//   substrings.size() == 3
//   substrings[0] == "apple"
//   substrings[1] == "orange"
//   substrings[2] == "banana"
//------------------------------------------------------------------------------

void SplitStringUsing(
    const std::string& full, const char* delim,
    std::vector<std::string>* result);

// This function has the same semnatic as SplitStringUsing.  Results
// are saved in an STL set container.
void SplitStringToSetUsing(
    const std::string& full, const char* delim, std::set<std::string>* result);

template <typename T>
struct simple_insert_iterator {
  explicit simple_insert_iterator(T* t) : t_(t) {}

  simple_insert_iterator<T>& operator=(const typename T::value_type& value) {
    t_->insert(value);
    return *this;
  }

  simple_insert_iterator<T>& operator*() { return *this; }
  simple_insert_iterator<T>& operator++() { return *this; }
  simple_insert_iterator<T>& operator++(int placeholder) { return *this; }

  T* t_;
};

template <typename T>
struct back_insert_iterator {
  explicit back_insert_iterator(T& t) : t_(t) {}

  back_insert_iterator<T>& operator=(const typename T::value_type& value) {
    t_.push_back(value);
    return *this;
  }

  back_insert_iterator<T>& operator*() { return *this; }
  back_insert_iterator<T>& operator++() { return *this; }
  back_insert_iterator<T> operator++(int placeholder) { return *this; }

  T& t_;
};

template <typename StringType, typename ITR>
static inline void SplitStringToIteratorUsing(
    const StringType& full, const char* delim, ITR* result) {
  CHECK_NOTNULL(delim);
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char* p = full.data();
    const char* end = p + full.size();
    while (p != end) {
      if (*p == c) {
        ++p;
      } else {
        const char* start = p;
        while (++p != end && *p != c) {
          // Skip to the next occurence of the delimiter.
        }
        *(*result)++ = StringType(start, p - start);
      }
    }
    return;
  }

  std::string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != std::string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *(*result)++ = full.substr(begin_index);
      return;
    }
    *(*result)++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

//------------------------------------------------------------------------------
// StringPrintf:
//
// For example:
//
//  std::string str = StringPrintf("%d", 1);    /* str = "1"  */
//  SStringPrintf(&str, "%d", 2);               /* str = "2"  */
//  StringAppendF(&str, "%d", 3);               /* str = "23" */
//------------------------------------------------------------------------------

std::string StringPrintf(const char* format, ...);
void SStringPrintf(std::string* dst, const char* format, ...);
void StringAppendF(std::string* dst, const char* format, ...);

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_COMMON_H_
