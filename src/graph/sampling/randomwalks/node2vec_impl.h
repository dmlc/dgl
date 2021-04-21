///*!
// *  Copyright (c) 2018 by Contributors
// * \file graph/sampling/node2vec_impl.cc
// * \brief DGL sampler - templated implementation definition of node2vec random walk.
// */
//
//#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_IMPL_H
//#define DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_IMPL_H
//
//#include <dgl/base_heterograph.h>
//#include <dgl/array.h>
//#include <vector>
//#include <utility>
//#include <functional>
//
//namespace dgl{
//
//    using namespace dgl::runtime;
//    using namespace dgl::aten;
//
//    namespace sampling{
//
//        namespace impl{
//
///*!
// * \brief Node2vec random walk.
// * \param g The graph
// * \param seeds A 1D array of seed nodes.
// * \param walk_length length of walk.
// * \param p transition probability
// * \param q transition probability
// * \param prob 1D float array, indicating the transition probability of each edge.
// *        An empty float array assumes uniform transition.
// */
//template<DLDeviceType XPU, typename IdxType>
//IdArray Node2vecRandomWalk(
//        const HeteroGraphPtr g,
//        const IdArray seeds,
//        const int64_t walk_length,
//        const double p,
//        const double q,
//        const FloatArray &prob);
//}
//
//}
//
//}
//#endif //DGL_NODE2VEC_RANDOMWALKS_IMPL_H
