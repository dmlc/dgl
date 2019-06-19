//
// Created by Allen Zhou on 2019/6/13.
//

#include <gtest/gtest.h>
#include <dgl/graph.h>
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dmlc/io.h>
#include <dgl/runtime/ndarray.h>

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dmlc::SeekStream;
using dgl::runtime::NDArray;

TEST(GraphTest, LoadDGLGraph){
  const char *filename = "data.dgl";
  int start_graph_idx = 1;
  int end_graph_idx = 10;

  SeekStream *fs = SeekStream::CreateForRead(filename);
  uint32_t magic_number, version, type;
  uint64_t section_offset;

  // Meta Data
  fs->Read(&magic_number);
  fs->Read(&version);
  fs->Read(&type);
  fs->Read(&section_offset);

  // Index Section
  fs->Seek(4096);
  uint32_t num_graphs;
  NDArray indices, num_nodes, num_edges, graph_labels;
  fs->Read(&num_graphs);
  indices.Load(fs);

  uint32_t *ind = static_cast<uint32_t *>(indices->data);

  uint32_t start_node = ind[start_graph_idx];
  uint32_t end_node = ind[end_graph_idx];








};