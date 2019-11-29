import itertools
import math
import numpy as np
import torch
import torch.nn.init as init
import torch.nn as nn

from ... import to_hetero
from ... import backend
from ...nn.pytorch import AtomicConv

class ParallelLinear(nn.Module):
    def __init__(self, num_channels, in_feats, out_feats, bias=True):
        super(ParallelLinear, self).__init__()

        self.num_channels = num_channels
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(num_channels, out_feats, in_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_channels, out_feats))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        for channel in range(self.num_channels):
            init.kaiming_uniform_(self.weight[channel], math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            for channel in range(self.num_channels):
                init.uniform_(self.bias[channel], -bound, bound)

    def forward(self, inputs):
        transformed_inputs = torch.einsum('fba, nfa->nfb', self.weight, inputs)
        if self.bias is not None:
            transformed_inputs = transformed_inputs + self.bias

        return transformed_inputs

class ACNNPredictor(nn.Module):
    """"""
    def __init__(self, in_size, hidden_sizes, features_to_use):
        super(ACNNPredictor, self).__init__()

        if features_to_use is None:
            self.num_channels = 1
        else:
            self.num_channels = len(features_to_use)
        self.features_to_use = features_to_use

        self.protein_project = self._build_projector(in_size, hidden_sizes)
        self.ligand_project = self._build_projector(in_size, hidden_sizes)
        self.complex_project = self._build_projector(in_size, hidden_sizes)

    def _build_projector(self, in_size, hidden_sizes):
        modules = []
        for h in hidden_sizes:
            modules.append(ParallelLinear(self.num_channels, in_size, h))
            modules.append(nn.ReLU())
            in_size = h
        modules.append(ParallelLinear(self.num_channels, in_size, 1))

        return nn.Sequential(*modules)

    def _finalize_features(self, conv_out, projector, node_feats):
        feats = projector(conv_out)
        if type(self.features_to_use) != type(None):
            mask = (node_feats == self.features_to_use).float().unsqueeze(-1)
            feats = feats * mask

        return feats.sum(dim=1)

    @staticmethod
    def sum_nodes(batch_size, batch_num_nodes, feats):
        """"""
        seg_id = torch.from_numpy(np.arange(batch_size, dtype='int64').repeat(batch_num_nodes))
        seg_id = seg_id.to(feats.device)

        return backend.unsorted_1d_segment_sum(feats, seg_id, batch_size, 0)

    def forward(self, protein_graph, ligand_graph, complex_graph,
                protein_conv_out, ligand_conv_out, complex_conv_out,
                protein_node_feats, ligand_node_feats, complex_node_feats):
        """
        Parameters
        ----------
        graph
        protein_conv_out : float32 tensor of shape (V1, K, F)
        ligand_conv_out : float32 tensor of shape (V2, K, F)
        complex_conv_out : float32 tensor of shape ((V1 + V2), K, F)
        protein_node_feats : float32 tensor of shape (V1, 1)
        ligand_node_feats : float32 tensor of shape (V2, 1)
        complex_node_feats : float32 tensor of shape ((V1 + V2), 1)
        """
        protein_feats = self._finalize_features(
            protein_conv_out, self.protein_project, protein_node_feats)
        ligand_feats = self._finalize_features(
            ligand_conv_out, self.ligand_project, ligand_node_feats)
        complex_feats = self._finalize_features(
            complex_conv_out, self.complex_project, complex_node_feats)

        protein_energy = self.sum_nodes(protein_graph.batch_size,
                                        protein_graph.batch_num_nodes,
                                        protein_feats)
        ligand_energy = self.sum_nodes(ligand_graph.batch_size,
                                       ligand_graph.batch_num_nodes,
                                       ligand_feats)

        with complex_graph.local_scope():
            complex_graph.ndata['h'] = complex_feats
            hetero_complex_graph = to_hetero(
                complex_graph, ntypes=complex_graph.original_ntypes,
                etypes=complex_graph.original_etypes)
            complex_energy_protein = self.sum_nodes(
                complex_graph.batch_size, complex_graph.batch_num_protein_nodes,
                hetero_complex_graph.nodes['protein_atom'].data['h'])
            complex_energy_ligand = self.sum_nodes(
                complex_graph.batch_size, complex_graph.batch_num_ligand_nodes,
                hetero_complex_graph.nodes['ligand_atom'].data['h'])
            complex_energy = complex_energy_protein + complex_energy_ligand

        return complex_energy - (protein_energy + ligand_energy)

class ACNN(nn.Module):
    """"""
    def __init__(self, hidden_sizes, features_to_use=None, radial=None):
        super(ACNN, self).__init__()

        if radial is None:
            radial = [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                      [0.0, 4.0, 8.0], [0.4]]
        radial_params = [x for x in itertools.product(*radial)]
        radial_params = torch.stack(list(map(torch.tensor, zip(*radial_params))), dim=1)

        self.protein_conv = AtomicConv(radial_params, features_to_use)
        self.ligand_conv = AtomicConv(radial_params, features_to_use)
        self.complex_conv = AtomicConv(radial_params, features_to_use)
        self.predictor = ACNNPredictor(radial_params.shape[0], hidden_sizes, features_to_use)

    def forward(self, graph):
        protein_graph = graph[('protein_atom', 'protein', 'protein_atom')]
        # Todo (Mufei): remove the two lines below after better built-in support
        protein_graph.batch_size = graph.batch_size
        protein_graph.batch_num_nodes = graph.batch_num_nodes('protein_atom')

        protein_graph_node_feats = protein_graph.ndata['atomic_number']
        assert protein_graph_node_feats.shape[-1] == 1
        protein_graph_distances = protein_graph.edata['distance']
        protein_conv_out = self.protein_conv(protein_graph,
                                             protein_graph_node_feats,
                                             protein_graph_distances)

        ligand_graph = graph[('ligand_atom', 'ligand', 'ligand_atom')]
        # Todo (Mufei): remove the two lines below after better built-in support
        ligand_graph.batch_size = graph.batch_size
        ligand_graph.batch_num_nodes = graph.batch_num_nodes('ligand_atom')

        ligand_graph_node_feats = ligand_graph.ndata['atomic_number']
        assert ligand_graph_node_feats.shape[-1] == 1
        ligand_graph_distances = ligand_graph.edata['distance']
        ligand_conv_out = self.ligand_conv(ligand_graph,
                                           ligand_graph_node_feats,
                                           ligand_graph_distances)

        complex_graph = graph[:, 'complex', :]
        # Todo (Mufei): remove the four lines below after better built-in support
        complex_graph.batch_size = graph.batch_size
        complex_graph.original_ntypes = graph.ntypes
        complex_graph.original_etypes = graph.etypes
        complex_graph.batch_num_protein_nodes = graph.batch_num_nodes('protein_atom')
        complex_graph.batch_num_ligand_nodes = graph.batch_num_nodes('ligand_atom')

        complex_graph_node_feats = complex_graph.ndata['atomic_number']
        assert complex_graph_node_feats.shape[-1] == 1
        complex_graph_distances = complex_graph.edata['distance']
        complex_conv_out = self.complex_conv(complex_graph,
                                             complex_graph_node_feats,
                                             complex_graph_distances)

        return self.predictor(
            protein_graph, ligand_graph, complex_graph,
            protein_conv_out, ligand_conv_out, complex_conv_out,
            protein_graph_node_feats, ligand_graph_node_feats, complex_graph_node_feats)
