import json
import numpy as np
import os
import tempfile
import torch
import pytest, unittest
import dgl
from dgl.data.utils import load_tensors, load_graphs
from chunk_graph import chunk_graph

from create_chunked_dataset import create_chunked_dataset

@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 4, 8])
def test_part_pipeline(num_chunks, num_parts):
    if num_chunks < num_parts:
        # num_parts should less/equal than num_chunks
        return

    with tempfile.TemporaryDirectory() as root_dir:
        all_ntypes, all_etypes, edge_data_gold = create_chunked_dataset(root_dir, num_chunks)
        
        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, '2parts')
        os.system('python3 tools/partition_algo/random_partition.py '\
                  '--in_dir {} --out_dir {} --num_partitions {}'.format(
                    in_dir, output_dir, num_parts))
        for ntype in ['author', 'institution', 'paper']:
            fname = os.path.join(output_dir, '{}.txt'.format(ntype))
            with open(fname, 'r') as f:
                header = f.readline().rstrip()
                assert isinstance(int(header), int)

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, 'parted_data')
        out_dir = os.path.join(root_dir, 'partitioned')
        ip_config = os.path.join(root_dir, 'ip_config.txt')
        with open(ip_config, 'w') as f:
            for i in range(num_parts):
                f.write(f'127.0.0.{i + 1}\n')

        cmd = 'python3 tools/dispatch_data.py'
        cmd += f' --in-dir {in_dir}'
        cmd += f' --partitions-dir {partition_dir}'
        cmd += f' --out-dir {out_dir}'
        cmd += f' --ip-config {ip_config}'
        cmd += ' --process-group-timeout 60'
        cmd += ' --save-orig-nids'
        cmd += ' --save-orig-eids'
        os.system(cmd)

        # check metadata.json
        meta_fname = os.path.join(out_dir, 'metadata.json')
        with open(meta_fname, 'rb') as f:
            meta_data = json.load(f)

        #all_etypes = ['affiliated_with', 'writes', 'cites', 'rev_writes']
        for etype in all_etypes:
            assert len(meta_data['edge_map'][etype]) == num_parts
        assert meta_data['etypes'].keys() == set(all_etypes)
        assert meta_data['graph_name'] == 'mag240m'

        #all_ntypes = ['author', 'institution', 'paper']
        for ntype in all_ntypes:
            assert len(meta_data['node_map'][ntype]) == num_parts
        assert meta_data['ntypes'].keys() == set(all_ntypes)
        assert meta_data['num_edges'] == g.num_edges()
        assert meta_data['num_nodes'] == g.num_nodes()
        assert meta_data['num_parts'] == num_parts

        # Create Id Map here.
        edge_dict = {
            "author:affiliated_with:institution": np.array([0, 2400]).reshape( #2400
                1, 2
            ),
            "author:writes:paper": np.array([2400, 16400]).reshape(1, 2), #12000
            "paper:cites:paper": np.array([16400, 40400]).reshape(1, 2), #24000
            "paper:rev_writes:author": np.array([40400, 52400]).reshape(1, 2),#12000
        }
        id_map = dgl.distributed.id_map.IdMap(edge_dict)
        etype_id, type_eid = id_map(np.arange(52400))

        for i in range(num_parts):
            sub_dir = 'part-' + str(i)
            assert meta_data[sub_dir]['node_feats'] == 'part{}/node_feat.dgl'.format(i)
            assert meta_data[sub_dir]['edge_feats'] == 'part{}/edge_feat.dgl'.format(i)
            assert meta_data[sub_dir]['part_graph'] == 'part{}/graph.dgl'.format(i)

            # check data
            sub_dir = os.path.join(out_dir, 'part' + str(i))

            # graph.dgl
            fname = os.path.join(sub_dir, 'graph.dgl')
            assert os.path.isfile(fname)
            g_list, data_dict = load_graphs(fname)
            part_g = g_list[0]
            assert isinstance(part_g, dgl.DGLGraph)

            # node_feat.dgl
            fname = os.path.join(sub_dir, 'node_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = ['paper/feat', 'paper/label', 'paper/year', 'paper/orig_ids']
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)
            ndata_paper_orig_ids = tensor_dict['paper/orig_ids']

            # orig_nids.dgl
            fname = os.path.join(sub_dir, 'orig_nids.dgl')
            assert os.path.isfile(fname)
            orig_nids = load_tensors(fname)
            assert len(orig_nids.keys()) == 3
            assert torch.equal(ndata_paper_orig_ids, orig_nids['paper'])

            # orig_eids.dgl
            fname = os.path.join(sub_dir, 'orig_eids.dgl')
            assert os.path.isfile(fname)
            orig_eids = load_tensors(fname)
            assert len(orig_eids.keys()) == 4

            # Read edge_feat.dgl
            fname = os.path.join(sub_dir, 'edge_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = [
                'paper:cites:paper/count',
                'author:writes:paper/year',
                'paper:rev_writes:author/year',
            ]
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)

            # Get orig_eids
            orig_type_eids = g.edata['orig_id'].numpy()
            orig_etype_ids = g.edata[dgl.ETYPE].numpy()

            # Compare the data stored as edge features in this partition with the data
            # from the original graph.
            etype_names = list(edge_dict.keys())
            for idx, etype_name in enumerate(etype_names):
                part_data = None
                key = None
                if etype_name + '/count' in tensor_dict:
                    key = etype_name + '/count'
                    part_data = tensor_dict[etype_name + '/count'].numpy()
                if etype_name + '/year' in tensor_dict:
                    key = etype_name + '/year'
                    part_data = tensor_dict[etype_name + '/year'].numpy()

                if part_data is None:
                    continue

                gold_type_ids = orig_type_eids[orig_etype_ids == idx]
                gold_data = edge_data_gold[key][gold_type_ids]
                assert np.all(gold_data == part_data)
