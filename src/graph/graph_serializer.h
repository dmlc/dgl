/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph_serializer.cc
 * \brief DGL serializer APIs
 */

#pragma once

#include <dgl/immutable_graph.h>
#include "heterograph.h"
#include "unit_graph.h"

namespace dgl {

class Serializer {
 public:
  static HeteroGraph* EmptyHeteroGraph() { return new HeteroGraph(); }
  static ImmutableGraph* EmptyImmutableGraph() {
    return new ImmutableGraph(static_cast<COOPtr>(nullptr));
  }
  static UnitGraph* EmptyUnitGraph() {
    return UnitGraph::EmptyGraph();
  }
};
}  // namespace dgl
