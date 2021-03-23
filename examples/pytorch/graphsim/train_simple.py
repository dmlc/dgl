from networkx.algorithms.connectivity import edge_augmentation
from models import MLP, OutputLayer,InteractionLayer
from dataloader import MultiBodyGraphCollator, MultiBodyTrainDataset,\
                       MultiBodyValidDataset, MultiBodyTestHalfDataset,\
                       MultiBodyTestFullDataset,MultiBodyTestDoubDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils.data import DataLoader
from functools import partial
import argparse

# Assuming no initial edge and global feature
class PrepareLayer(nn.Module):
    def __init__(self,node_feats,stat):
        super(PrepareLayer,self).__init__()
        self.node_feats = node_feats
        # stat {'median':median,'max':max,'min':min}
        self.stat = stat

    def normalize_input(self,node_feature):
        return (node_feature-self.stat['median'])*(2/(self.stat['max']-self.stat['min']))

    def forward(self,g,node_feature):
        with g.local_scope():
            node_feature = self.normalize_input(node_feature)
            g.ndata['feat'] = node_feature[:,1:] # Only dynamic feature
            g.apply_edges(fn.u_sub_v('feat','feat','e'))
            edge_feature = g.edata['e']
            return node_feature,edge_feature

class IN(nn.Module):
    def __init__(self,node_feats,stat):
        super(IN,self).__init__()
        self.node_feats = node_feats
        self.stat = stat
        edge_fc = partial(MLP,num_layers=5,hidden=150)
        node_fc = partial(MLP,num_layers=2,hidden=100)

        self.in_layer = InteractionLayer(node_feats,
                                         node_feats-1,
                                         out_node_feats=2,
                                         out_edge_feats=50,
                                         edge_fc=edge_fc,
                                         node_fc=node_fc,
                                         mode='e')
    
    # Denormalize Velocity only
    def denormalize_output(self,out):
        return out*(self.stat['max'][3:5]-self.stat['min'][3:5])/2+self.stat['median'][3:5]

    def forward(self,g,n_feat,e_feat):
        with g.local_scope():
            out_n,out_e = self.in_layer(g,n_feat,e_feat)
            out_n = self.denormalize_output(out_n)
            return out_n,out_e

def train(optimizer,loss_fn,model,prep,dataloader,gamma):
    total_loss = 0
    for i,(graph_batch,data_batch,label_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        node_feat,edge_feat = prep(graph_batch,data_batch)
        v_pred,out_e = model(graph_batch,node_feat.float(),edge_feat.float())
        loss = loss_fn(v_pred,label_batch)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        loss = loss + gamma*torch.norm(out_e,p=2)
        loss.backward()
        optimizer.step()
        #print("Batch: ",i)
    return total_loss
# Need to calculate second order loss
def eval(loss_fn,model,prep,dataloader):
    total_loss = 0
    for i,(graph_batch,data_batch,label_batch) in enumerate(dataloader):
        node_feat,edge_feat = prep(graph_batch,data_batch)
        v_pred,_ = model(graph_batch,node_feat.float(),edge_feat.float())
        loss = loss_fn(v_pred,label_batch)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        #print("Batch: ",i)
    return total_loss

def eval_rollout():
    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr',type=float,default=0.001)
    argparser.add_argument('--epochs',type=int,default=40000)
    argparser.add_argument('--gamma',type=float,default=0.001)
    args = argparser.parse_args()


    train_data = MultiBodyTrainDataset()
    valid_data = MultiBodyValidDataset()
    test_full_data = MultiBodyTestFullDataset()
    test_half_data = MultiBodyTestHalfDataset()
    test_doub_data = MultiBodyTestDoubDataset()
    collator = MultiBodyGraphCollator(train_data.n_particles)
    half_collator = MultiBodyGraphCollator(train_data.n_particles//2)
    doub_collator = MultiBodyGraphCollator(train_data.n_particles*2)

    train_dataloader = DataLoader(train_data,100,True,collate_fn=collator)
    valid_dataloader = DataLoader(valid_data,100,True,collate_fn=collator)
    test_full_dataloader = DataLoader(test_full_data,100,True,collate_fn=collator)
    test_half_dataloader = DataLoader(test_half_data,100,True,collate_fn=half_collator)
    test_doub_dataloader = DataLoader(test_doub_data,100,True,collate_fn=doub_collator)

    node_feats = 5
    stat = {'median':torch.from_numpy(train_data.stat_median),
            'max':torch.from_numpy(train_data.stat_max),
            'min':torch.from_numpy(train_data.stat_min)}

    prepare_layer = PrepareLayer(node_feats,stat)
    interaction_net = IN(node_feats,stat)

    optimizer = torch.optim.Adam(interaction_net.parameters(),lr=args.lr,weight_decay=args.gamma)
    loss_fn = torch.nn.MSELoss()
    import time
    for e in range(args.epochs):
        last_t = time.time()
        loss = train(optimizer,loss_fn,interaction_net,prepare_layer,train_dataloader,args.gamma)
        print("Epoch time: ",time.time()-last_t)
        if e %100 ==0:
            valid_loss = eval(loss_fn,interaction_net,prepare_layer,valid_dataloader)
            test_full_loss = eval(loss_fn,interaction_net,prepare_layer,test_full_dataloader)
            test_half_loss = eval(loss_fn,interaction_net,prepare_layer,test_half_dataloader)
            test_doub_loss = eval(loss_fn,interaction_net,prepare_layer,test_doub_dataloader)
            print("Epoch: {},Loss: Valid: {} Full: {} Half: {} Double: {}".format(e,valid_loss,
                                                                                  test_full_data,
                                                                                  test_half_loss,
                                                                                  test_doub_loss))






