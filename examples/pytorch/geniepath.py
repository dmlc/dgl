#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:34:38 2018

@author: ivabruge

GeniePath: Graph Neural Networks with Adaptive Receptive Paths
Paper: https://arxiv.org/abs/1802.00910

this model uses an LSTM on the node reductions of the message-passing step 

we store the network states at the graph node, since the LSTM variables are not transmitted

"""

from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import load_data, preprocess_features

class NodeReduceModule(nn.Module):
    def __init__(self, input_dim, num_hidden, num_heads=3, input_dropout=None,
            attention_dropout=None, act=lambda x: F.softmax(F.leaky_relu(x), dim=0)):
        super(NodeReduceModule, self).__init__()
        self.num_heads = num_heads
        self.input_dropout = input_dropout
        self.attention_dropout = attention_dropout
        self.act = act
        self.fc = nn.ModuleList(
                [nn.Linear(input_dim, num_hidden, bias=False)
                    for _ in range(num_heads)])
        self.attention = nn.ModuleList(
                [nn.Linear(num_hidden * 2, 1, bias=False) for _ in range(num_heads)])

    def forward(self, msgs):
        src, dst = zip(*msgs)
        hu = torch.cat(src, dim=0) # neighbor repr
        hv = torch.cat(dst, dim=0)

        msgs_repr = []

        # iterate for each head
        for i in range(self.num_heads):
            # calc W*hself and W*hneigh
            hvv = self.fc[i](hv)
            huu = self.fc[i](hu)
            # calculate W*hself||W*hneigh
            h = torch.cat((hvv, huu), dim=1)
            a = self.act(self.attention[i](h))
            if self.attention_dropout is not None:
                a = F.dropout(a, self.attention_dropout)
            if self.input_dropout is not None:
                hvv = F.dropout(hvv, self.input_dropout)
            h = torch.sum(a * hvv, 0, keepdim=True)
            msgs_repr.append(h)

        return msgs_repr


class NodeUpdateModule(nn.Module):
    def __init__(self, residual, fc, act, aggregator):
        super(NodeUpdateModule, self).__init__()
        
        self.residual = residual
        self.fc = fc
        self.act = act
        self.aggregator = aggregator

    def forward(self, node, msgs_repr):
        # apply residual connection and activation for each head
        for i in range(len(msgs_repr)):
            if self.residual:
                h = self.fc[i](node['h'])
                msgs_repr[i] = msgs_repr[i] + h
            if self.act is not None:
                msgs_repr[i] = self.act(msgs_repr[i])

        # aggregate multi-head results
        h = self.aggregator(msgs_repr)
        c0 = torch.zeros(h.shape)
        if node['c'] is None:
            c0 = torch.zeros(h.shape)
        else:
            c0 = node['c']
        if node['h_i'] is None:
            h0 = torch.zeros(h.shape)
        else:
            h0 = node['h_i']
        lstm = nn.LSTM(input_size=h.shape[1], hidden_size=h.shape[1], num_layers=1)
        
        #add dimension to handle sequential (create sequence of length 1)
        h, (h_i, c) = lstm(h.unsqueeze(0), (h0.unsqueeze(0), c0.unsqueeze(0)))
        
        #remove sequential dim
        h = torch.squeeze(h, 0)
        h_i = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        
        return {'h': h, 'c':c, 'h_i':h_i}

class GeniePath(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, num_heads,
            activation, input_dropout, attention_dropout, use_residual=False ):
        super(GeniePath, self).__init__()

        self.input_dropout = input_dropout
        self.reduce_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        # hidden layers
        for i in range(num_layers):
            if i == 0:
                last_dim = in_dim
                residual = False
            else:
                last_dim = num_hidden * num_heads # because of concat heads
                residual = use_residual
            self.reduce_layers.append(
                    NodeReduceModule(last_dim, num_hidden, num_heads, input_dropout,
                        attention_dropout))
            self.update_layers.append(
                    NodeUpdateModule(residual, self.reduce_layers[-1].fc, activation,
                        lambda x: torch.cat(x, 1)))
        # projection
        self.reduce_layers.append(
            NodeReduceModule(num_hidden * num_heads, num_classes, 1, input_dropout,
                attention_dropout))
        self.update_layers.append(
            NodeUpdateModule(False, self.reduce_layers[-1].fc, None, sum))

    def forward(self, g):
        g.register_message_func(lambda src, dst, edge: (src['h'], dst['h']))
        for reduce_func, update_func in zip(self.reduce_layers, self.update_layers):
            # apply dropout
            if self.input_dropout is not None:
                # TODO (lingfan): use batched dropout once we have better api
                #                 for global manipulation
                for n in g.nodes():
                    g.node[n]['h'] = F.dropout(g.node[n]['h'], p=self.input_dropout)
                    g.node[n]['c'] = None
                    g.node[n]['h_i'] = None
            g.register_reduce_func(reduce_func)
            g.register_update_func(update_func)
            g.update_all()
        logits = [g.node[n]['h'] for n in g.nodes()]
        logits = torch.cat(logits, dim=0)
        return logits
    
    #train on graph g with features, and target labels. Accepts a loss function and an optimizer function which implements optimizer.step()
    def train(self, g, features, labels, epochs, loss_f=torch.nn.NLLLoss, loss_params={}, optimizer=torch.optim.Adam, optimizer_parameters=None, lr=0.001, ignore=[0], quiet=False):
        
        labels = torch.LongTensor(labels)
        _, labels = torch.max(labels, dim=1)
        # convert labels and masks to tensor
        
        if optimizer_parameters is None:
            optimizer_parameters = self.parameters()
            
        #instantiate optimizer on given params
        optimizer_f = optimizer(optimizer_parameters, lr)        
        
        for epoch in range(args.epochs):
            # reset grad
            optimizer_f.zero_grad()
    
            # reset graph states
            for n in g.nodes():
                g.node[n]['h'] = torch.FloatTensor(features[n].toarray())
    
            # forward
            logits = self.forward(g)
   
            #intantiate loss on passed parameters (e.g. class weight params)         
            loss = loss_f(**loss_params)
            
            #trim null labels
            idx = [i for i, a in enumerate(labels) if a not in ignore]
            logits = logits[idx, :]
            labels = labels[idx]
            out = loss(logits, labels)
            
            if not quiet:
                print("epoch {} loss: {}".format(epoch, out))
                
            out.backward()
            optimizer_f.step()

def main(args):
    # dropout parameters
    input_dropout = args.idrop
    attention_dropout = args.adrop

    # load and preprocess dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    features = preprocess_features(features)

    # initialize graph
    g = DGLGraph(adj)

    # create model
    model = GeniePath(args.num_layers,
                features.shape[1],
                args.num_hidden,
                y_train.shape[1],
                args.num_heads,
                F.elu,
                input_dropout,
                attention_dropout,
                args.residual)
    model.train(g, features, y_train, epochs=args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset name")
    parser.add_argument("--epochs", type=int, default=10,
            help="training epoch")
    parser.add_argument("--num-heads", type=int, default=3,
            help="number of attentional heads to use")
    parser.add_argument("--num-layers", type=int, default=1,
            help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
            help="size of hidden units")
    parser.add_argument("--residual", action="store_true",
            help="use residual connection")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--idrop", type=float, default=0.2,
            help="Input dropout")
    parser.add_argument("--adrop", type=float, default=0.2,
            help="attention dropout")
    
    args = parser.parse_args()
    print(args)

    main(args)