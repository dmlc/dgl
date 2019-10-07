/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/sampler.h
 * \brief DGL sampler header.
 */

#ifndef DGL_GRAPH_SAMPLER_RANDOMWALK_H_
#define DGL_GRAPH_SAMPLER_RANDOMWALK_H_

#include <dgl/runtime/object.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <memory>

namespace dgl {

namespace sampling {

/*! \brief Structure of multiple random walk traces */
struct RandomWalkTraces : public runtime::Object {
  /*! \brief number of traces generated for each seed */
  IdArray trace_counts;
  /*! \brief length of each trace, concatenated */
  IdArray trace_lengths;
  /*! \brief the vertices, concatenated */
  IdArray vertices;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("vertices", &vertices);
    v->Visit("trace_lengths", &trace_lengths);
    v->Visit("trace_counts", &trace_counts);
  }

  static constexpr const char *_type_key = "sampler.RandomWalkTraces";
  DGL_DECLARE_OBJECT_TYPE_INFO(RandomWalkTraces, runtime::Object);
};
typedef std::shared_ptr<RandomWalkTraces> RandomWalkTracesPtr;
DGL_DEFINE_OBJECT_REF(RandomWalkTracesRef, RandomWalkTraces);

/*!
 * \brief Batch-generate random walk traces
 * \param seeds The array of starting vertex IDs
 * \param num_traces The number of traces to generate for each seed
 * \param num_hops The number of hops for each trace
 * \return a flat ID array with shape (num_seeds, num_traces, num_hops + 1)
 */
IdArray RandomWalk(const GraphInterface *gptr,
                   IdArray seeds,
                   int num_traces,
                   int num_hops);

/*!
 * \brief Batch-generate random walk traces with restart
 *
 * Stop generating traces if max_frequrent_visited_nodes nodes are visited more than
 * max_visit_counts times.
 *
 * \param seeds The array of starting vertex IDs
 * \param restart_prob The restart probability
 * \param visit_threshold_per_seed Stop generating more traces once the number of nodes
 * visited for a seed exceeds this number.  (Algorithm 1 in [1])
 * \param max_visit_counts Alternatively, stop generating traces for a seed if no less
 * than \c max_frequent_visited_nodes are visited no less than \c max_visit_counts
 * times.  (Algorithm 2 in [1])
 * \param max_frequent_visited_nodes See \c max_visit_counts
 * \return A RandomWalkTraces pointer.
 *
 * \sa [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
 */
RandomWalkTracesPtr RandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes);

/*
 * \brief Batch-generate random walk traces with restart on a bipartite graph, walking two
 * hops at a time.
 *
 * Since it is walking on a bipartite graph, the vertices of a trace will always stay on the
 * same side.
 *
 * Stop generating traces if max_frequrent_visited_nodes nodes are visited more than
 * max_visit_counts times.
 *
 * \param seeds The array of starting vertex IDs
 * \param restart_prob The restart probability
 * \param visit_threshold_per_seed Stop generating more traces once the number of nodes
 * visited for a seed exceeds this number.  (Algorithm 1 in [1])
 * \param max_visit_counts Alternatively, stop generating traces for a seed if no less
 * than \c max_frequent_visited_nodes are visited no less than \c max_visit_counts
 * times.  (Algorithm 2 in [1])
 * \param max_frequent_visited_nodes See \c max_visit_counts
 * \return A RandomWalkTraces instance.
 *
 * \note Doesn't verify whether the graph is indeed a bipartite graph
 *
 * \sa [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
 */
RandomWalkTracesPtr BipartiteSingleSidedRandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes);

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLER_RANDOMWALK_H_
