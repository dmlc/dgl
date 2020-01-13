namespace dgl {

namespace sampling {

namespace impl {

template
std::pair<IdArray, TypeArray> RandomWalkImpl<kDLCPU>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const FloatArray prob) {
  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = etypes->shape[0] + 1;

  IdArray vids = IdArray::Empty(
    {num_seeds, trace_length}, seeds->dtype, seeds->ctx);
  TypeArray vtypes = TypeArray::Empty(
    {num_seeds, trace_length}, etypes->dtype, etypes->ctx);

#pragma omp parallel for
  for (int64_t i = 0; i < num_seeds; ++i) {
    IdArray vids_i = vids.CreateView(
      {trace_length}, vids->dtype, i * trace_length * vids->dtype.bits / 8);
    TypeArray vtypes_i = vtypes.CreateView(
      {trace_length}, vtypes->dtype, i * trace_length * vtypes->dtype.bits / 8);

    RandomWalkOneSeed(hg, IndexSelect(seeds, i), etypes, prob, vids_i, vtypes_i, 0.);
  }

  return std::make_pair(vids, vtypes);
}

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
