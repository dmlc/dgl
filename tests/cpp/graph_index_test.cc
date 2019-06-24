/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph_index_test.cc
 * \brief Test GraphIndex
 */
#include <gtest/gtest.h>
#include <dgl/graph.h>

TEST(GraphTest, TestNumVertices){
  dgl::Graph g(false);
  g.AddVertices(10);
  ASSERT_EQ(g.NumVertices(), 10);
};
