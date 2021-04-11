import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
import utils


from utils import thread_wrapped_func
from load_graph import load_reddit, inductive_split

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def producer(q, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2, train_nfeat, train_labels, feat_dimension, label_dimension, device):
    th.cuda.set_device(device)

    # Map input tensors into GPU address
    train_nfeat = train_nfeat.to(device="unified")
    train_labels = train_labels.to(device="unified")

    # Create GPU-side ping pong buffers
    in_feat1 = th.zeros(feat_dimension, device=device)
    in_feat2 = th.zeros(feat_dimension, device=device)
    in_label1 = th.zeros(label_dimension, dtype=th.long, device=device)
    in_label2 = th.zeros(label_dimension, dtype=th.long, device=device)

    # Termination signal
    finish = th.ones(1, dtype=th.bool)

    # Share with the training process
    q.put((in_feat1, in_feat2, in_label1, in_label2, finish))
    print("Allocation done")

    flag = 1

    with th.no_grad():
        while(1):
            event1.wait()
            event1.clear()
            if not finish:
                break
            if flag:
                th.index_select(train_nfeat, 0, idxf1[0:idxf1_len].to(device=device), out=in_feat1[0:idxf1_len])
                th.index_select(train_labels, 0, idxl1[0:idxl1_len].to(device=device), out=in_label1[0:idxl1_len])
            else:
                th.index_select(train_nfeat, 0, idxf2[0:idxf2_len].to(device=device), out=in_feat2[0:idxf2_len])
                th.index_select(train_labels, 0, idxl2[0:idxl2_len].to(device=device), out=in_label2[0:idxl2_len])
            flag = (flag == False)
            th.cuda.synchronize()
            event2.set()

#### Entry point

def run(q, args, device, data, in_feats, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2):
    th.cuda.set_device(device)

    # Unpack data
    n_classes, train_g, val_g, test_g = data

    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    in_feat1, in_feat2, in_label1, in_label2, finish = q.get()

    # A prologue for the pipelining purpose, just for the first minibatch of the first epoch
    # ------------------------------------------------------
    flag = True
    input_nodes, seeds, blocks_next = next(iter(dataloader))

    # Send node indices for the next minibatch to the producer
    if flag:
        idxf1[0:len(input_nodes)].copy_(input_nodes)
        idxl1[0:len(seeds)].copy_(seeds)
        idxf1_len.fill_(len(input_nodes))
        idxl1_len.fill_(len(seeds))
    else:
        idxf2[0:len(input_nodes)].copy_(input_nodes)
        idxl2[0:len(seeds)].copy_(seeds)
        idxf2_len.fill_(len(input_nodes))
        idxl2_len.fill_(len(seeds))
    event1.set()
    time.sleep(1)

    input_nodes_n = len(input_nodes)
    seeds_n = len(seeds)
    flag = (flag == False)
    blocks_temp = blocks_next
    # ------------------------------------------------------
    # Prologue done

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks_next) in enumerate(dataloader):
            tic_step = time.time()

            # Send node indices for the next minibatch to the producer
            if flag:
                idxf1[0:len(input_nodes)].copy_(input_nodes)
                idxl1[0:len(seeds)].copy_(seeds)
                idxf1_len.fill_(len(input_nodes))
                idxl1_len.fill_(len(seeds))
            else:
                idxf2[0:len(input_nodes)].copy_(input_nodes)
                idxl2[0:len(seeds)].copy_(seeds)
                idxf2_len.fill_(len(input_nodes))
                idxl2_len.fill_(len(seeds))

            event1.set()

            event2.wait()
            event2.clear()

            # Load the input features as well as output labels
            if not flag:
                batch_inputs = in_feat1[0:input_nodes_n]
                batch_labels = in_label1[0:seeds_n]
            else:
                batch_inputs = in_feat2[0:input_nodes_n]
                batch_labels = in_label2[0:seeds_n]

            blocks = [block.int().to(device) for block in blocks_temp]
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flag = (flag == False)
            input_nodes_n = len(input_nodes)
            seeds_n = len(seeds)
            blocks_temp = blocks_next

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))

        # A prologue for the next epoch
        # ------------------------------------------------------
        input_nodes, seeds, blocks_next = next(iter(dataloader))

        if flag:
            idxf1[0:len(input_nodes)].copy_(input_nodes)
            idxl1[0:len(seeds)].copy_(seeds)
            idxf1_len.fill_(len(input_nodes))
            idxl1_len.fill_(len(seeds))
        else:
            idxf2[0:len(input_nodes)].copy_(input_nodes)
            idxl2[0:len(seeds)].copy_(seeds)
            idxf2_len.fill_(len(input_nodes))
            idxl2_len.fill_(len(seeds))
        event1.set()

        event2.wait()
        event2.clear()

        # Load the input features as well as output labels
        if not flag:
            batch_inputs = in_feat1[0:input_nodes_n]
            batch_labels = in_label1[0:seeds_n]
        else:
            batch_inputs = in_feat2[0:input_nodes_n]
            batch_labels = in_label2[0:seeds_n]

        # Compute loss and prediction
        blocks = [block.int().to(device) for block in blocks_temp]
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        flag = (flag == False)
        input_nodes_n = len(input_nodes)
        seeds_n = len(seeds)
        blocks_temp = blocks_next
        # ------------------------------------------------------
        # Prologue done

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(
                model, val_g, val_nfeat, val_labels, val_nid, device)
            test_acc = evaluate(
                model, test_g, test_nfeat, test_labels, test_nid, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    # Send a termination signal to the producer
    finish.copy_(th.zeros(1, dtype=th.bool))
    event1.set()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--mps', type=str, default='0')
    args = argparser.parse_args()

    device = th.device('cuda:%d' % args.gpu)
    mps = list(map(str, args.mps.split(',')))

    # If MPS values are given, then setup MPS
    if float(mps[0]) != 0:
        user_id = utils.mps_get_user_id()
        utils.mps_daemon_start()
        utils.mps_server_start(user_id)
        server_pid = utils.mps_get_server_pid()
        time.sleep(4)

    g, n_classes = load_reddit()
    # Construct graph
    g = dgl.as_heterograph(g)

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    data = n_classes, train_g, val_g, test_g

    train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features').share_memory_()
    train_labels = val_labels = test_labels = g.ndata.pop('labels').share_memory_()
    in_feats = train_nfeat.shape[1]

    fanout_max = 1
    for fanout in args.fan_out.split(','):
        fanout_max = fanout_max * int(fanout)

    feat_dimension = [args.batch_size * fanout_max, train_nfeat.shape[1]]
    label_dimension = [args.batch_size]

    ctx = mp.get_context('spawn')

    if float(mps[0]) != 0:
        utils.mps_set_active_thread_percentage(server_pid, mps[0])
        # Just in case we add a timer to make sure MPS setup is done before we launch producer
        time.sleep(4)

    # TODO: shared structure declarations can be futher simplified
    q = ctx.SimpleQueue()

    # Synchornization signals
    event1 = ctx.Event()
    event2 = ctx.Event()

    # Indices and the their lengths shared between the producer and the training processes
    idxf1 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxf2 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxl1 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxl2 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxf1_len = th.zeros([1], dtype=th.long).share_memory_()
    idxf2_len = th.zeros([1], dtype=th.long).share_memory_()
    idxl1_len = th.zeros([1], dtype=th.long).share_memory_()
    idxl2_len = th.zeros([1], dtype=th.long).share_memory_()

    print("Producer Start")
    producer_inst = ctx.Process(target=producer,
                    args=(q, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2, train_nfeat, train_labels, feat_dimension, label_dimension, device))
    producer_inst.start()

    if float(mps[0]) != 0:
        # Just in case we add timers to make sure MPS setup is done before we launch training
        time.sleep(8)
        utils.mps_set_active_thread_percentage(server_pid, mps[1])
        time.sleep(4)

    print("Run Start")
    p = mp.Process(target=thread_wrapped_func(run),
                    args=(q, args, device, data, in_feats, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2))
    p.start()

    p.join()
    producer_inst.join()
