"""
Model module for LightGCN.
Contains the LightGCN model and BPRLoss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_size, n_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                dglnn.GraphConv(emb_size, emb_size, weight=False, bias=False)
            )
        # Initialize user and item node embeddings with a normal (Gaussian) distribution.
        self.embedding_user = nn.Embedding(n_users, emb_size)
        self.embedding_item = nn.Embedding(n_items, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding_user.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.embedding_item.weight, mean=0.0, std=0.1)

    def forward(self, g):
        h_user = self.embedding_user.weight.to(g.device)
        h_item = self.embedding_item.weight.to(g.device)
        # Store embeddings from all layers
        embeddingList_user = [h_user]
        embeddingList_item = [h_item]
        for i, layer in enumerate(self.layers):
            h_all = torch.cat([h_user, h_item], dim=0)
            h_all_new = layer(g, h_all)
            h_user = h_all_new[:self.n_users]
            h_item = h_all_new[self.n_users:]
            embeddingList_user.append(h_user)
            embeddingList_item.append(h_item)
        # Take the embedding of each layer and compute the average (the core idea of LightGCN)
        final_user_emb = torch.mean(torch.stack(embeddingList_user, dim=1), dim=1)
        final_item_emb = torch.mean(torch.stack(embeddingList_item, dim=1), dim=1)
        # Concatenate user and item embeddings, and return the complete embedding
        all_emb = torch.cat([final_user_emb, final_item_emb], dim=0)
        return all_emb
    
    def bprLoss(self, user_emb, pos_emb, neg_emb, batch_user, batch_pos, batch_neg):
        pos_score = torch.mul(user_emb, pos_emb)
        pos_score = torch.sum(pos_score, dim=1)
        neg_score = torch.mul(user_emb, neg_emb)
        neg_score = torch.sum(neg_score, dim=1)
        loss = torch.mean(F.softplus(neg_score - pos_score))
        # Compute the RegLoss
        userEmb0 = self.embedding_user.weight[batch_user]
        posEmb0 = self.embedding_item.weight[batch_pos]
        negEmb0 = self.embedding_item.weight[batch_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(batch_user))
        return loss.cpu(), reg_loss.cpu()