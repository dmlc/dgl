/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>
#include <dgl/graph.h>

TEST(GraphTest, TestNumVertices){
  dgl::Graph g(false);
  g.AddVertices(10);
  ASSERT_EQ(g.NumVertices(), 10);
};
