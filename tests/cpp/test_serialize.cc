#include <dmlc/memory_io.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "../../src/graph/unit_graph.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::aten;
using namespace dmlc;

TEST(Serialize, UnitGraph) {
  aten::CSRMatrix csr_matrix;
  auto src = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = VecToIdArray<int64_t>({1, 6, 2, 6});
  auto mg = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst);
  UnitGraph* ug = dynamic_cast<UnitGraph*>(mg.get());
  std::string blob;
  dmlc::MemoryStringStream ifs(&blob);

  static_cast<dmlc::Stream*>(&ifs)->Write<UnitGraph>(*ug);

  dmlc::MemoryStringStream ofs(&blob);
  src = NewIdArray(0);
  dst = NewIdArray(0);
  auto mg2 = dgl::UnitGraph::CreateFromCOO(
      1, 0, 0, src, dst);  // Any way to construct Empty UnitGraph?
  UnitGraph* ug2 = dynamic_cast<UnitGraph*>(mg2.get());
  static_cast<dmlc::Stream*>(&ofs)->Read(ug2);
  EXPECT_EQ(ug2->NumVertices(0), 8);
  EXPECT_EQ(ug2->NumVertices(1), 9);
  EXPECT_EQ(ug2->NumEdges(0), 4);
}