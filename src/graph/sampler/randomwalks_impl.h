/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/cpu/randomwalks.h
 * \brief DGL sampler templated implementation of random walks
 */

namespace dgl {

namespace sampling {

namespace impl {

template<DLDeviceType XPU, typename Idx, typename Type>
std::pair<IdArray, TypeArray> RandomWalkOnce(
    const HeteroGraphPtr hg,
    Idx seed,
    const TypeArray etypes,
    const FloatArray prob,
    IdArray vids,
    IdArray vtypes) {
  int64_t num_seeds = seeds->shape[0];
  int64_t num_etypes = etypes->shape[0];
  // TODO
}

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

