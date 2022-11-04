/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/meta_utils.h
 * \brief Meta programming utils
 * \author Pawel Piotrowicz <pawel.piotrowicz@intel.com>
 */
#ifndef INTEL_META_UTILS_H_
#define INTEL_META_UTILS_H_
#include <tuple>

namespace dgl {
namespace utils {

template <typename T, typename Tuple>
struct has_type;

template <typename T>
struct has_type<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

template <
    class OCmp, template <class> class ToP, class Tup,
    int ok = std::tuple_size<Tup>::value>
struct DeepType;

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 1> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  enum { value = std::is_same<OCmp, ToP<EL1>>::value };
};

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 2> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  typedef typename std::tuple_element<1, Tup>::type EL2;
  enum {
    value =
        (std::is_same<OCmp, ToP<EL1>>::value ||
         std::is_same<OCmp, ToP<EL2>>::value)
  };
};

template <class OCmp, template <class> class ToP, class Tup>
struct DeepType<OCmp, ToP, Tup, 3> {
  typedef typename std::tuple_element<0, Tup>::type EL1;
  typedef typename std::tuple_element<1, Tup>::type EL2;
  typedef typename std::tuple_element<2, Tup>::type EL3;
  enum {
    value =
        (std::is_same<OCmp, ToP<EL1>>::value ||
         std::is_same<OCmp, ToP<EL2>>::value ||
         std::is_same<OCmp, ToP<EL3>>::value)
  };
};

template <bool b>
using Required = typename std::enable_if<b, bool>::type;

template <class L, class R>
using CheckCmp = Required<std::is_same<L, R>::value>;

template <class L, class R1, class R2>
using CheckCmp_2 =
    Required<std::is_same<L, R1>::value || std::is_same<L, R2>::value>;

template <class OpType, template <class> class TPP, class Tup>
using Verify = Required<utils::DeepType<OpType, TPP, Tup>::value>;

}  // namespace utils
}  // namespace dgl
#endif  //  INTEL_META_UTILS_H_
