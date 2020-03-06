
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_message(edges):
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


class LinearLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, features):
        features_output = self.linear(features)
        features_output = self.activation(features_output)
        return features_output


class AtomToPair(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(AtomToPair, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, atom_features, atom_to_pair):
        AP_ij = self.linear(tf.gather(atom_features, atom_to_pair))
        AP_ij = self.activation(AP_ij)

        AP_ji = self.linear(tf.gather(atom_features, tf.reverse(atom_to_pair, [1])))
        AP_ji = self.activation(AP_ji)

        return AP_ij + AP_ji


class PairToAtom(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(PairToAtom, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, pair_features, pair_split):
        PA = self.linear(pair_features)
        PA = self.activation(PA)
        PA = tf.segment_sum(PA, pair_split)

        return PA


class WeaveModule(nn.Module):

    def __init__(self,
                 n_atom_input_feat=75,
                 n_pair_input_feat=14,
                 n_atom_output_feat=50,
                 n_pair_output_feat=50,
                 n_hidden_AA=50,
                 n_hidden_PA=50,
                 n_hidden_AP=50,
                 n_hidden_PP=50,
                 update_pair=True):  # update_pair means apply updates on w and b of pairs (A_P, P_P, P)
        super(WeaveModule, self).__init__()
        n_hidden_A = n_hidden_AA + n_hidden_PA
        n_hidden_P = n_hidden_AP + n_hidden_PP
        self.update_pair = update_pair
        self.atom_layer = LinearLayer(n_hidden_A, n_atom_output_feat, F.relu)
        self.pair_layer = LinearLayer(n_hidden_P, n_pair_output_feat, F.relu)
        self.atom_to_atom = LinearLayer(n_atom_input_feat, n_hidden_AA, F.relu)
        self.pair_to_pair = LinearLayer(n_pair_input_feat, n_hidden_PP, F.relu)
        self.atom_to_pair = AtomToPair(n_atom_input_feat * 2, n_hidden_AP, F.relu)
        self.pair_to_atom = PairToAtom(n_pair_input_feat, n_hidden_PA, F.relu)

    def forward(self, g, inputs, atom_only=False):
        """
        inputs: [atom_features, pair_features, atom_to_pair, pair_split]
        """

        atom_x = inputs[0]
        pair_x = inputs[1]
        atom_to_pair = inputs[2]
        pair_split = inputs[3]

        g.ndata['h'] = atom_x
        # g.edata['m'] = pair_x  # pair is not edge
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        atom_x = g.ndata.pop['h']

        a0 = self.atom_to_atom(atom_x)
        a1 = self.pair_to_atom(pair_x, pair_split)
        a = tf.concat([a0, a1], axis=1)
        next_atom = self.atom_layer(a)
        if atom_only:
            return next_atom

        p0 = self.atom_to_pair(atom_x, atom_to_pair)
        p1 = self.pair_to_pair(pair_x)
        p = tf.concat([p0, p1], axis=1)
        next_pair = self.pair_layer(p)
        return next_atom, next_pair
