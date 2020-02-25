#include <dgl/immutable_graph.h>
#include <dmlc/memory_io.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "../../src/graph/graph_serializer.h"
#include "../../src/graph/heterograph.h"
#include "../../src/graph/unit_graph.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::aten;
using namespace dmlc;

TEST(Serialize, DISABLED_UnitGraph) {
  aten::CSRMatrix csr_matrix;
  auto src = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = VecToIdArray<int64_t>({1, 6, 2, 6});
  auto mg = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst);
  UnitGraph* ug = dynamic_cast<UnitGraph*>(mg.get());
  std::string blob;
  dmlc::MemoryStringStream ifs(&blob);

  static_cast<dmlc::Stream*>(&ifs)->Write<UnitGraph>(*ug);

  dmlc::MemoryStringStream ofs(&blob);
  UnitGraph* ug2 = Serializer::EmptyUnitGraph();
  static_cast<dmlc::Stream*>(&ofs)->Read(ug2);
  EXPECT_EQ(ug2->NumVertices(0), 9);
  EXPECT_EQ(ug2->NumVertices(1), 8);
  EXPECT_EQ(ug2->NumEdges(0), 4);
  EXPECT_EQ(ug2->FindEdge(0, 1).first, 2);
  EXPECT_EQ(ug2->FindEdge(0, 1).second, 6);
  delete ug2;
}

TEST(Serialize, DISABLED_ImmutableGraph) {
  auto src = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = VecToIdArray<int64_t>({1, 6, 2, 6});
  auto gptr = ImmutableGraph::CreateFromCOO(10, src, dst);
  ImmutableGraph* rptr = gptr.get();

  std::string blob;
  dmlc::MemoryStringStream ifs(&blob);

  static_cast<dmlc::Stream*>(&ifs)->Write(*rptr);

  dmlc::MemoryStringStream ofs(&blob);
  ImmutableGraph* rptr_read = new ImmutableGraph(static_cast<COOPtr>(nullptr));
  static_cast<dmlc::Stream*>(&ofs)->Read(rptr_read);
  EXPECT_EQ(rptr_read->NumEdges(), 4);
  EXPECT_EQ(rptr_read->NumVertices(), 10);
  EXPECT_EQ(rptr_read->FindEdge(2).first, 5);
  EXPECT_EQ(rptr_read->FindEdge(2).second, 2);
  delete rptr_read;
}

TEST(Serialize, DISABLED_HeteroGraph) {
  auto src = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = VecToIdArray<int64_t>({1, 6, 2, 6});
  auto mg1 = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst);
  src = VecToIdArray<int64_t>({6, 2, 5, 1, 8});
  dst = VecToIdArray<int64_t>({5, 2, 4, 8, 0});
  auto mg2 = dgl::UnitGraph::CreateFromCOO(1, 9, 9, src, dst);
  std::vector<HeteroGraphPtr> relgraphs;
  relgraphs.push_back(mg1);
  relgraphs.push_back(mg2);
  src = VecToIdArray<int64_t>({0, 0});
  dst = VecToIdArray<int64_t>({1, 0});
  auto meta_gptr = ImmutableGraph::CreateFromCOO(2, src, dst);
  HeteroGraph* hrptr = new HeteroGraph(meta_gptr, relgraphs);

  std::string blob;
  dmlc::MemoryStringStream ifs(&blob);
  static_cast<dmlc::Stream*>(&ifs)->Write(*hrptr);

  dmlc::MemoryStringStream ofs(&blob);
  HeteroGraph* gptr = dgl::Serializer::EmptyHeteroGraph();
  static_cast<dmlc::Stream*>(&ofs)->Read(gptr);
  EXPECT_EQ(gptr->NumVertices(0), 9);
  EXPECT_EQ(gptr->NumVertices(1), 8);
  delete hrptr;
  delete gptr;
}
