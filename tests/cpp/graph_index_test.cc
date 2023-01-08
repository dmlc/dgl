/**
 *  Copyright (c) 2019 by Contributors
 * @file graph_index_test.cc
 * @brief Test GraphIndex
 */
#include <dgl/graph.h>
#include <gtest/gtest.h>

TEST(GraphTest, TestNumVertices) {
  dgl::Graph g;
  g.AddVertices(10);
  ASSERT_EQ(g.NumVertices(), 10);
};
