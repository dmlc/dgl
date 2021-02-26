import numpy as np
import torch as th
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
from qm9_v2 import QM9Dataset_v2
from model import InfoGraphS
import argparse


def argument():
    parser = argparse.ArgumentParser(description='InfoGraphS')

    # data source params
    parser.add_argument('--target', type=str, default='mu', help='Choose regression task}')
    parser.add_argument('--train_num', type=int, default=5000, help='Number of training set')

    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Training batch size.')
    parser.add_argument('--val_batch_size', type=int, default=100, help='Validating batch size.')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='weight decay.')

    # model params
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden layer dimensionalities')
    parser.add_argument('--reg', type=int, default=0.001, help='regularization coefficent')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    return args


def collate(samples):
    ''' collate function for building graph dataloader '''

    # generate batched graphs and labels
    graphs, targets = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_targets = th.Tensor(targets)
    
    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)
    
    batched_graph.ndata['graph_id'] = graph_id
    
    return batched_graph, batched_targets


if __name__ == '__main__':

    # Step 1: Prepare graph data   ===================================== #
    args = argument()
    label_keys = [args.target]
    print(args)

    dataset = QM9Dataset_v2(label_keys)
    dataset.to_dense()

    graphs = dataset.graphs

    # Train/Val/Test Splitting
    N = len(graphs)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)

    val_num = 10000
    test_num = 10000

    val_idx = all_idx[:val_num]
    test_idx = all_idx[val_num : val_num + test_num]
    train_idx = all_idx[val_num + test_num : val_num + test_num + args.train_num]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    unsup_idx = all_idx[val_num + test_num:]
    unsup_data = [dataset[i] for i in unsup_idx]

    # generate supervised training dataloader and unsupervised training dataloader
    train_loader = GraphDataLoader(train_data,
                                   batch_size=args.batch_size,
                                   collate_fn=collate,
                                   drop_last=False,
                                   shuffle=True)

    unsup_loader = GraphDataLoader(unsup_data,
                                   batch_size=args.batch_size,
                                   collate_fn=collate,
                                   drop_last=False,
                                   shuffle=True)

    # generate validation & testing datalaoder

    val_loader = GraphDataLoader(val_data,
                                 batch_size=args.val_batch_size,
                                 collate_fn=collate,
                                 drop_last=False,
                                 shuffle=True)

    test_loader = GraphDataLoader(test_data,
                                  batch_size=args.val_batch_size,
                                  collate_fn=collate,
                                  drop_last=False,
                                  shuffle=True)

    print('======== target = {} ========'.format(args.target))

    mean = dataset.labels.mean().item()
    std = dataset.labels.mean().item()


    print('mean = {:4f}'.format(mean))
    print('std = {:4f}'.format(std))

    in_dim = dataset[0][0].ndata['attr'].shape[1]

    # Step 2: Create model =================================================================== #
    model = InfoGraphS(in_dim, args.hid_dim)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001
    )

    # Step 4: training epoches =============================================================== #
    sup_loss_all = 0
    unsup_loss_all = 0
    consis_loss_all = 0

    best_val_error = 99999
    best_test_error = 99999
    for epoch in range(args.epochs):
        ''' Training '''
        model.train()
        lr = scheduler.optimizer.param_groups[0]['lr']

        iteration = 0
        sup_loss_all = 0
        unsup_loss_all = 0
        consis_loss_all = 0

        for sup_data, unsup_data in zip(train_loader, unsup_loader):
            sup_graph, sup_target = sup_data
            unsup_graph, _ = unsup_data

            sup_graph = sup_graph.to(args.device)
            unsup_graph = unsup_graph.to(args.device)

            sup_target = (sup_target - mean) / std
            sup_target = sup_target.to(args.device)

            optimizer.zero_grad()

            sup_loss = F.mse_loss(model(sup_graph), sup_target)
            unsup_loss, consis_loss = model.unsup_forward(unsup_graph)

            loss = sup_loss + unsup_loss + args.reg * consis_loss

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            consis_loss_all += consis_loss.item()

            optimizer.step()

        print('Epoch: {}, Sup_Loss: {:4f}, Unsup_loss: {:.4f}, Consis_loss: {:.4f}' \
            .format(epoch, sup_loss_all, unsup_loss_all, consis_loss_all))

        model.eval()

        val_error = 0
        test_error = 0
        

        for val_graphs, val_targets in val_loader:

            val_graph = val_graphs.to(args.device)
            val_target = (val_targets - mean) / std
            val_target = val_target.to(args.device)

            val_error += (model(val_graph) * std - val_target * std).abs().sum().item()

        val_error = val_error / val_num
        scheduler.step(val_error)

        if val_error < best_val_error:
            best_val_error = val_error

            for test_graphs, test_targets in test_loader:

                test_graph = test_graphs.to(args.device)
                test_target = (test_targets - mean) / std
                test_target = test_target.to(args.device)

                test_error += (model(test_graph) * std - test_target * std).abs().sum().item()

            test_error = test_error / test_num
            best_test_error = test_error

        print('Epoch: {}, LR: {}, best_val_error: {:.4f}, val_error: {:.4f}, best_test_error: {:.4f}' \
            .format(epoch, lr, best_val_error, val_error, best_test_error))
