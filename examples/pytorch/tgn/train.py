import dgl
import torch
import numpy as np
from tgn import TGN
from data_preprocess import TemporalWikipediaDataset,TemporalRedditDataset,TemporalDataset
from dataloading import FastTemporalEdgeCollator,NegativeSampler,FastTemporalSampler,TemporalEdgeDataLoader,temporal_sort,TemporalSampler,TemporalEdgeCollator
import argparse
import traceback
import time

from sklearn.metrics import average_precision_score, roc_auc_score

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)

def train(model,dataloader,sampler,criterion,optimizer,batch_size,fast_mode):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    for _,positive_pair_g,negative_pair_g,blocks in dataloader:
        optimizer.zero_grad()
        pred_pos,pred_neg = model.embed(positive_pair_g,negative_pair_g,blocks)
        loss = criterion(pred_pos,torch.ones_like(pred_pos))
        loss+= criterion(pred_neg,torch.zeros_like(pred_neg))
        total_loss += float(loss)*batch_size
        retain_graph = True if batch_cnt == 0 and not fast_mode else False
        loss.backward(retain_graph = retain_graph)
        optimizer.step()
        model.detach_memory()
        model.update_memory(positive_pair_g)
        if fast_mode:
            sampler.attach_last_update(model.memory.last_update_t)
        print("Batch: ",batch_cnt,"Time: ",time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss


def test_val(model,dataloader,sampler,criterion,batch_size,fast_mode):
    model.eval()
    batch_size = batch_size
    total_loss = 0
    aps,aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for _,postive_pair_g,negative_pair_g,blocks in dataloader:
            pred_pos, pred_neg = model.embed(postive_pair_g,negative_pair_g,blocks)
            loss = criterion(pred_pos,torch.ones_like(pred_pos))
            loss+= criterion(pred_neg,torch.zeros_like(pred_neg))
            total_loss += float(loss)*batch_size
            y_pred = torch.cat([pred_pos,pred_neg],dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)),torch.zeros(pred_neg.size(0))],dim=0)
            model.update_memory(postive_pair_g)
            if fast_mode:
                sampler.attach_last_update(model.memory.last_update_t)
            aps.append(average_precision_score(y_true,y_pred))
            aucs.append(roc_auc_score(y_true,y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",type=int,default=50,help='epochs for training on entire dataset')
    parser.add_argument("--batch_size",type=int,default=200,help="Size of each batch")
    parser.add_argument("--embedding_dim",type=int,default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim",type=int,default=100,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim",type=int,default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater",type=str,default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator",type=str,default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors",type=int,default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method",type=str,default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads",type=int,default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode",action="store_true",default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--num_negative_samples",type=int,default=1,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset",type=str,default=1,
                        help="dataset selection wikipedia/reddit")

    args = parser.parse_args()
    if args.dataset == 'wikipedia':
        data = TemporalWikipediaDataset()
    elif args.dataset == 'reddit':
        data = TemporalRedditDataset()
    else:
        print("Warning Using Untested Dataset: "+args.dataset)
        data = TemporalDataset(args.dataset)

    # Pre-process data, mask new node in test set from original graph
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    bidir_graph = dgl.add_reverse_edges(data,copy_edata=True)
    num_edges = data.num_edges()
    trainval_div = int(VALID_SPLIT*num_edges)

    # Select new node from test set and remove them from entire graph
    test_nodes = torch.cat([data.edges()[0][trainval_div:],data.edges()[1][trainval_div:]]).unique().numpy()
    test_mask  = np.random.choice(test_nodes,int(0.1*len(test_nodes)),replace=False)
    test_mask = set(data.nodes().numpy())-set(test_mask)
    test_mask = np.array(list(test_mask))
    graph_remove_node = dgl.node_subgraph(data,test_mask)
    # Need to sort the graph for sampling order
    graph_remove_node = temporal_sort(graph_remove_node,'timestamp')
    # Remove orphan nodes
    graph_no_new_node = dgl.compact_graphs([graph_remove_node])[0]
    data.ndata[dgl.NID] = data.nodes()
    graph_new_node = data
    bidir_graph_no_new_node = dgl.add_reverse_edges(graph_no_new_node,copy_edata=True)


    # Sampler Initialization
    if args.fast_mode:
        sampler = FastTemporalSampler(data,k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data,k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    no_new_node_neg_sampler = NegativeSampler(k=args.num_negative_samples,g = graph_no_new_node)
    new_node_neg_sampler    = NegativeSampler(k=args.num_negative_samples,g = data)
    # Set Train, validation, test and new node test id
    num_edges_no_new_node = graph_no_new_node.num_edges()
    train_seed = torch.arange(int(TRAIN_SPLIT*num_edges_no_new_node))
    valid_seed = torch.arange(int(TRAIN_SPLIT*num_edges_no_new_node),int(VALID_SPLIT*num_edges_no_new_node))
    test_seed  = torch.arange(int(VALID_SPLIT*num_edges_no_new_node),num_edges_no_new_node)
    test_new_node_seed = torch.arange(int(VALID_SPLIT*num_edges),num_edges)

    g_sampling = None if args.fast_mode else bidir_graph_no_new_node
    new_node_g_sampling = None if args.fast_mode else bidir_graph
    new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    train_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                        train_seed,
                                        sampler,
                                        batch_size = args.batch_size,
                                        negative_sampler=no_new_node_neg_sampler,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8,
                                        collator=edge_collator,
                                        g_sampling=g_sampling)

    valid_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                        valid_seed,
                                        sampler,
                                        batch_size = args.batch_size,
                                        negative_sampler=no_new_node_neg_sampler,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8,
                                        collator=edge_collator,
                                        g_sampling = g_sampling)

    test_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                        test_seed,
                                        sampler,
                                        batch_size = args.batch_size,
                                        negative_sampler=no_new_node_neg_sampler,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8,
                                        collator=edge_collator,
                                        g_sampling = g_sampling)

    test_new_node_dataloader = TemporalEdgeDataLoader(graph_new_node,
                                        test_new_node_seed,
                                        new_node_sampler if args.fast_mode else sampler,
                                        batch_size = args.batch_size,
                                        negative_sampler=new_node_neg_sampler,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8,
                                        collator=edge_collator,
                                        g_sampling=new_node_g_sampling)

    

    edge_dim = data.edata['feats'].shape[1]
    num_node = data.num_nodes()

    model = TGN(edge_feat_dim = edge_dim,
                memory_dim = args.memory_dim,
                temporal_dim = args.temporal_dim,
                embedding_dim= args.embedding_dim,
                num_heads = args.num_heads,
                num_nodes = num_node,
                n_neighbors = args.n_neighbors,
                memory_updater_type=args.memory_updater)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(model.parameters(),lr=0.0001)
    # Implement Logging mechanism
    f = open("logging.txt",'w')
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_loss     = train(model,train_dataloader,sampler,criterion,optimizer,args.batch_size,args.fast_mode)
            val_ap,val_auc   = test_val(model,valid_dataloader,sampler,criterion,args.batch_size,args.fast_mode)
            memory_checkpoint = model.store_memory()
            test_ap,test_auc = test_val(model,test_dataloader,sampler,criterion,args.batch_size,args.fast_mode)
            model.restore_memory(memory_checkpoint)
            if args.fast_mode:
                new_node_sampler.sync(sampler)
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap,nn_test_auc = test_val(model,test_new_node_dataloader,sample_nn,criterion,args.batch_size,args.fast_mode)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(i,train_loss,val_ap,val_auc))
            log_content.append("Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i,test_ap,test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(i,nn_test_ap,nn_test_auc))


            f.writelines(log_content)
            model.reset_memory()
            if i < args.epochs-1 and args.fast_mode:
                sampler.reset()
            print(log_content[0],log_content[1],log_content[2])
    except :
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
        #exit(-1)
    print("========Training is Done========")