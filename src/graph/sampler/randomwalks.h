namespace dgl {

namespace sampling {

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const FloatArray prob);

};  // namespace sampling

};  // namespace dgl
