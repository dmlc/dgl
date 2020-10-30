#-*- coding:utf-8 -*-


import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl


class NodeExplainerModule(nn.Module):
    """
    A Pytorch module for explaining a node's prediction based on its computational graph and node features.
    Use two masks: One mask on edges, and another on nodes' features.

    So far due to the limit of DGL on edge mask operation, this explainer need the to-be-explained models to
    accept an additional input argument, edge mask, and apply this mask in their inner message parse operation.

    !!!!!!......This is the current walk_around......!!!!!!
    """

    # Class inner variables
    loss_coef = {
        "g_size": 0.005,
        "feat_size": 1.0,
        "g_ent": 1.0,
        "feat_ent": 0.1,
        "lap": 0.0
    }

    # Variables of results for calling

    def __init__(self,
                 model,
                 num_nodes,
                 node_feat_dim,
                 activation='sigmoid',
                 agg_fn='sum',
                 verbos=False):
        super(NodeExplainerModule, self).__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.activation = activation
        self.agg_fn=agg_fn
        self.verbose = verbos

        # Initialize parameters on masks
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_nodes)
        self.node_feat_mask = self.create_node_feat_mask(self.node_feat_dim)


    def create_edge_mask(self, num_edges, init_strategy='normal', const=1.):
        """
        Based on the number of nodes in the computational graph, create a learnable mask of edges.

        To adopt to DGL, change this mask from N*N adjacency matrix to the No. of edges

        Parameters
        ----------
        num_edges: Integer N, specify the number of edges.
        init_strategy: String, specify the parameter initializati　on method
        const: Float, a value for constant initialization

        Returns
        -------
        mask and mask bias: Tensor, all in shape of N*1

        """
        mask = nn.Parameter(th.Tensor(num_edges, 1), dtype=th.float)

        if init_strategy == 'normal':
            std = nn.init.calculate_gain("relu") * th.sqrt(
                1.0 / num_edges
            )
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1), dtype=th.float)
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias


    def creat_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """
        Based on the dimensions of node feature in the computational graph, create a learnable mask of features.

        Parameters
        ----------
        node_feat_dim: Integer N, dimensions of node feature
        init_strategy: String, specify the parameter initialization method

        Returns
        -------
        mask: Tensor, in shape of N
        """
        mask = nn.Parameter(node_feat_dim, dtype=th.float)

        if init_strategy == "normal":
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask


    def reset(self):
        """
        Reset inner parameters for retrain.

        Returns
        -------

        """
        pass

    def forward(self, graph, n_idx, n_feats):
        """
        Calculate predict results after masking input of the given model.

        This version will

        Parameters
        ----------
        graph: DGLGraph, Should be a sub_graph of the target node to be explained.
        n_idx: Tensor, an integer, index of the node to be explained.
        pred_logits:

        Returns
        -------

        """
        # Extract related features
        num_nodes = graph.num_of_nodes()

        # Step 1: Mask node feature with the inner feature mask
        new_n_feats = n_feats * self.node_feat_mask.sigmoid()
        self.edge_mask = self.edge_mask.sigmoid()

        # Step 2:
        new_logits = self.model(graph, new_n_feats, self.edge_mask)

        return new_logits


    def _loss(self, pred_logits, pred_label):
        """
        Compute the losses of this explainer, which include 6 parts in author's codes:
        1. The prediction loss between predict logits before and after node and edge masking;
        2. Loss of edge mask itself, which tries to put the mask value to either 0 or 1;
        3. Loss of node feature mask itself,  which tries to put the mask value to either 0 or 1;
        4. L2 loss of edge mask weights, but in sum not in mean;
        5. L2 loss of node feature mask weights, which is NOT used in the author's codes;
        6. Laplacian loss of the adj matrix.

        In the PyG implementation, there are 5 types of losses:
        1. The prediction loss between logits before and after node and edge masking;
        2. Sum loss of edge mask weights;
        3. Loss of edge mask entropy, which tries to put the mask value to either 0 or 1;
        4. Sum loss of node feature mask weights;
        5. Loss of node feature mask entropy, which tries to put the mask value to either 0 or 1;

        In this implementation, will use the PyG's version, but set the aggregation function of weights
        to be optional.

        Parameters
        ----------
        pred_logits：Tensor, N-dim logits output of model
        pred_label: Tensor, N-dim one-hot label of the label

        Returns
        -------
        loss: Scalar, the overall loss of this explainer.

        """
        # 1. prediction loss
        log_logit = th.log_softmax(pred_logits)
        pred_loss = th.sum(log_logit * pred_label)

        # 2. edge mask loss
        if self.activation == 'sigmoid':
            edge_mask = F.sigmoid(self.edge_mask)
        elif self.activation == 'relu':
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        edge_mask_loss = self.loss_coef['g_size'] * th.sum(edge_mask)

        # 3. edge mask entropy loss
        edge_ent = -edge_mask * \
                   th.log(edge_mask + 1e-8) - \
                   (1 - edge_mask) * \
                   th.log(1 - edge_mask + 1e-8)
        edge_ent_loss = self.loss_coef['g_ent'] * th.mean(edge_ent)

        # 4. node feature mask loss
        if self.activation == 'sigmoid':
            node_feat_mask = F.sigmoid(self.node_feat_mask)
        elif self.activation == 'relu':
            node_feat_mask = F.relu(self.node_feat_mask)
        else:
            raise ValueError()
        node_feat_mask_loss = self.loss_coef['feat_size'] * th.sum(node_feat_mask)

        # 5. node feature mask entry loss
        node_feat_ent = -node_feat_mask * \
                        th.log(node_feat_mask + 1e-8) - \
                        (1 - node_feat_mask) * \
                        th.log( 1 - node_feat_mask + 1e-8)
        node_feat_ent_loss = self.loss_coef['feat_ent'] * th.mean(node_feat_ent)

        total_loss = pred_loss + edge_mask_loss + edge_ent_loss + node_feat_mask_loss + node_feat_ent_loss

        return total_loss
