import dgl
import unittest

@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')

def test_degree_balanced_dataloader():
    from dgl.dataloading import DegreeBalancedDataloader
    import torch
    g = dgl.graph(([1, 2, 3, 4, 0, 3, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6]))
    nids = torch.arange(g.number_of_nodes())
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # not shuffle
    dataloader = DegreeBalancedDataloader(
        g, nids, sampler, max_node=4, max_degree=4,
        shuffle=False, device="cpu", num_workers=0)
    for input_nodes, output_nodes, blocks in dataloader:
        print(blocks, input_nodes, output_nodes)
        assert(blocks[0].num_dst_nodes() <= 4)
        assert(blocks[0].num_edges() <= 4)
    # max_node is None
    print()
    dataloader = DegreeBalancedDataloader(
        g, nids, sampler, max_node=None, max_degree=4,
        shuffle=False, device="cpu", num_workers=0)
    for input_nodes, output_nodes, blocks in dataloader:
        print(blocks, input_nodes, output_nodes)
        assert(blocks[0].num_edges() <= 4)
    # shuffle
    print()
    dataloader = DegreeBalancedDataloader(
        g, nids, sampler, max_node=4, max_degree=4,
        shuffle=True, device="cpu", num_workers=0)
    for input_nodes, output_nodes, blocks in dataloader:
        print(blocks, input_nodes, output_nodes)
        assert(blocks[0].num_dst_nodes() <= 4)
        assert(blocks[0].num_edges() <= 4)

if __name__ == '__main__':
    test_degree_balanced_dataloader()
