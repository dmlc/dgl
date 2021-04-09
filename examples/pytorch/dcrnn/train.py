from functools import partial
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from model import DCRNN, DiffConv, GatedGAT
from dataloading import METR_LAGraphDataset,METR_LATrainDataset,\
                        METR_LATestDataset,METR_LAValidDataset,\
                        PEMS_BAYGraphDataset,PEMS_BAYTrainDataset,\
                        PEMS_BAYValidDataset,PEMS_BAYTestDataset
from utils import NormalizationLayer, masked_mse_loss, get_learning_rate

# TODO: Define the training logic

def train(model,graph,dataloader,optimizer,scheduler,normalizer,loss_fn,device,minimum_lr,max_grad_norm):
    # The model input will be first normialized and denormalize when computing the loss
    total_loss = []
    # The batch size is fixed as well as how should we use the graph hence we need to 
    # Do dynamic padding.
    graph = graph.to(device)
    for i,(x,y) in enumerate(dataloader):
        optimizer.zero_grad()
        x_norm = normalizer.normalize(x).view(x.shape[1],-1,x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).view(x.shape[1],-1,x.shape[3]).float().to(device)
        y = y.view(y.shape[1],-1,y.shape[3]).float().to(device)
        # No need to permute data but need to handle the batched graph
        batch_size = x.shape[0]
        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph,x_norm,y_norm,i,device)
        # Denormalization for loss compute
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred,y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > minimum_lr:
            scheduler.step()
        total_loss.append(float(loss))
        print("Batch {}".format(i))
    return total_loss

def eval(model,graph,dataloader,normalizer,loss_fn,device):
    total_loss = []
    graph = graph.to(device)
    with torch.no_grad():
        for i,(x,y) in enumerate(dataloader):
            x_norm = normalizer.normalize(x).view(x.shape[1],-1,x.shape[3]).float().to(device)
            y_norm = normalizer.normalize(y).view(x.shape[1],-1,x.shape[3]).float().to(device)
            y = y.view(x.shape[1],-1,x.shape[3]).to(device)
            batch_size = x.shape[0]
            batch_graph = dgl.batch([graph]*batch_size)
            output = model(batch_graph,x_norm,y_norm,i,device)
            y_pred = normalizer.denormalize(output)
            loss = loss_fn(y_pred,y)
            total_loss.append(float(loss))
        return np.mean(total_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--model',type=str,default='diffconv')
    parser.add_argument('--gpu',type=int,default=-1)
    parser.add_argument('--diffsteps', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--decay_steps', type=int, default=2000)
    parser.add_argument('--lr',type=float, default = 0.01)
    parser.add_argument('--minimum_lr',type=float,default=2e-6)
    parser.add_argument('--dataset',type=str,default='LA',help='BAY')
    parser.add_argument('--epochs', type= int, default=1000)
    parser.add_argument('--max_grad_norm',type=float,default=5.0)
    args = parser.parse_args()
    # Load the datasets
    if args.dataset == 'LA':
        g = METR_LAGraphDataset()
        train_data = METR_LATrainDataset()
        test_data = METR_LATestDataset()
        valid_data = METR_LAValidDataset()
    elif args.dataset == 'BAY':
        g = PEMS_BAYGraphDataset()
        train_data = PEMS_BAYTrainDataset()
        test_data = PEMS_BAYTestDataset()
        valid_data = PEMS_BAYValidDataset()

    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    train_loader = DataLoader(train_data,batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    valid_loader = DataLoader(valid_data,batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    test_loader = DataLoader(test_data,batch_size=args.batch_size,num_workers = args.num_workers, shuffle = True)
    normalizer = NormalizationLayer(train_data.mean,train_data.std)

    net = partial(DiffConv,k=args.diffsteps) if args.model == 'diffconv' else partial(GatedGAT,map_feats = 64, num_heads=args.num_heads)

    dcrnn = DCRNN(in_feats = 2,
                  out_feats = 64,
                  seq_len=12,
                  num_layers=2,
                  net=net,
                  decay_steps=args.decay_steps).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    loss_fn = masked_mse_loss

    for e in range(args.epochs):
        train_loss = train(dcrnn,g,train_loader,optimizer,scheduler,normalizer,loss_fn,device,args.minimum_lr,args.max_grad_norm)
        valid_loss = eval(dcrnn,g,valid_loader,normalizer,loss_fn,device)
        test_loss = eval(dcrnn,g,test_loader,normalizer,loss_fn,device)
        print("Epoch: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(e,
                                                                             train_loss/len(train_data),
                                                                             valid_loss/len(valid_data),
                                                                             test_loss/len(test_data)))

