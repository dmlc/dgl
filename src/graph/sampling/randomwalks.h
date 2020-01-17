/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks.h
 * \brief DGL sampler - templated implementation definition of random walks
 */

#ifndef SAMPLER_RANDOMWALKS_H_
#define SAMPLER_RANDOMWALKS_H_

#include <dgl/runtime/container.h>
#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <utility>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<DLDeviceType XPU>
std::pair<IdArray, TypeArray> RandomWalkImpl(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const List<Value> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // SAMPLER_RANDOMWALKS_H_
