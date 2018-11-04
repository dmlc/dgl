/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/traversal.h
 * \brief Graph traversal routines.
 *
 * All traversal rountines are edge frontier generators. Each frontier is a
 * list of edges (specified by edge ids). An optional tag can be specified
 * on each edge (represented by an int value).
 */
#ifndef DGL_TRAVERSAL_H_
#define DGL_TRAVERSAL_H_

#include "graph.h"

namespace dgl {

/*!
 * \brief Class for edge frontiers.
 *
 * Each frontier is a
 * list of edges (specified by edge ids). An optional tag can be specified
 * on each edge (represented by an int value).
 */
class Frontiers {
 public:
  /*!
   * \brief default constructor
   * \param has_tags If true, edges will have tags.
   */
  explicit Frontiers(bool has_tags): has_tags_(has_tags) {}

  /*! \brief default destructor */
  ~Frontiers() = default;

  /*! \brief default copy constructor */
  Frontiers(const Frontiers& other) = default;

  /*! \brief default move constructor */
  Frontiers(Frontiers&& other) = default;

  /*! \brief Start a new frontier */
  void NewFrontier();

  /*!
   * \brief Add one edge to the current frontier
   * \param eid The edge id
   * \param tag The optional edge tag
   */
  void AddEdge(dgl_id_t eid, int tag = -1);

  /*! \brief getter for the edge id vector */
  const std::vector<dgl_id_t>& edge_ids() const { return edge_ids_; }

  /*! \brief getter for the edge tag vector */
  const std::vector<int64_t>& edge_tags() const { return edge_tags_; }

  /*! \brief getter for the frontier section vector */
  const std::vector<int64_t>& sections() const { return sections_; }

 private:
  /*!\brief a flag of whether tags are required */
  const bool has_tags_;

  /*!\brief a vector store for the edges in all the fronties */
  std::vector<dgl_id_t> edge_ids_;

  /*!\brief a vector store for edge tags */
  std::vector<int64_t> edge_tags_;

  /*!\brief a section vector to indicate each frontier */
  std::vector<int64_t> sections_;
};

/*!
 * \brief Produce frontiers in a breadth-first-search (BFS) order.
 *
 * \param source Source nodes.
 * \param reversed If true, BFS follows the in-edge direction
 * \return the frontiers
 */
Frontiers BFS(IdArray source, bool reversed);

}  // namespace dgl

#endif  // DGL_TRAVERSAL_H_
