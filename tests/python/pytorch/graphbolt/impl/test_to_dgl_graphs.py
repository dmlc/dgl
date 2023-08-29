import dgl.graphbolt as gb
import dgl
import torch


def test_to_dgl_graphs_hetero():
  relation = ('A', 'relation', 'B')
  node_pairs = {relation: (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))}
  reverse_column_node_ids = {'B': torch.tensor([10, 11, 12, 13, 14, 16])}
  reverse_row_node_ids = {'A': torch.tensor([5, 9, 7]), 'B': torch.tensor([10, 11, 12, 13, 14, 16])}
  reverse_edge_ids = {relation: torch.tensor([19, 20, 21])}
  subgraph = gb.SampledSubgraphImpl(
      node_pairs=node_pairs,
      reverse_column_node_ids=reverse_column_node_ids,
      reverse_row_node_ids=reverse_row_node_ids,
      reverse_edge_ids=reverse_edge_ids
  )

  g = gb.DataBlock(sampled_subgraphs=[subgraph]).to_dgl_graphs()[0]
  assert torch.equal(g.edges()[0], node_pairs[relation][0])
  assert torch.equal(g.edges()[1], node_pairs[relation][1])
  assert torch.equal(g.ndata[dgl.NID]['A'], reverse_row_node_ids['A'])
  assert torch.equal(g.ndata[dgl.NID]['B'], reverse_row_node_ids['B'])
  assert torch.equal(g.edata[dgl.EID], reverse_edge_ids[relation])


def test_to_dgl_graphs_homo():
  node_pairs = (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))
  reverse_column_node_ids = torch.tensor([10, 11, 12])
  reverse_row_node_ids = torch.tensor([10, 11, 12, 13, 14, 16])
  reverse_edge_ids = torch.tensor([19, 20, 21])
  subgraph = gb.SampledSubgraphImpl(
      node_pairs=node_pairs,
      reverse_column_node_ids=reverse_column_node_ids,
      reverse_row_node_ids=reverse_row_node_ids,
      reverse_edge_ids=reverse_edge_ids
  )
  g = gb.DataBlock(sampled_subgraphs=[subgraph]).to_dgl_graphs()[0]
  
  assert torch.equal(g.edges()[0], node_pairs[0])
  assert torch.equal(g.edges()[1], node_pairs[1])
  assert torch.equal(g.ndata[dgl.NID], reverse_row_node_ids)
  assert torch.equal(g.edata[dgl.EID], reverse_edge_ids)