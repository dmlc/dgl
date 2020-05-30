"""
Graph Attention Networks (PPI Dataset) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code implements
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import numpy as np
import dgl
import time
import tensorflow as tf
import argparse
from sklearn.metrics import f1_score
from model import GCN
from dgl.data.ppi import PPIDataset
from torch.utils.data import DataLoader

def collate(sample):
    graphs, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs, node_attrs='feat')
    feats = graph.ndata['feat']
    labels = tf.constant(np.concatenate(labels), dtype=tf.float32)
    degs = tf.cast(tf.identity(graph.in_degrees()), dtype=tf.float32)
    norm = tf.math.pow(degs, -0.5)
    norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)

    graph.ndata['norm'] = tf.expand_dims(norm, -1)
    return graph, feats, labels

def evaluate(feats, model, subgraph, labels, loss_fcn):
    
    model.g = subgraph
    # for layer in model.layers:
    #     layer.g = subgraph
    output = model(feats, training=False)
    loss_data = loss_fcn(labels, output)
    predict = np.where(output.numpy() >= 0., 1, 0)
    score = f1_score(labels.numpy(),
                        predict, average='micro')
    return score, loss_data.numpy().item()
        
def main(args):
    if args.gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)

    batch_size = args.batch_size
    cur_step = 0
    patience = args.patience
    best_score = -1
    best_loss = 10000
    # define loss function
    loss_fcn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)
    # create the dataset
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # define the model    
    model = GCN(g,
                num_feats,
                args.num_hidden,
                n_classes,
                args.num_layers,
                tf.nn.relu,
                args.in_drop)
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=args.lr)
    total_dur = []
    for epoch in range(args.epochs):
        loss_list = []
        dur = []
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            model.g = subgraph
            # for layer in model.layers:
            #     layer.g = subgraph
            if epoch >= 3:
                t0 = time.time()
            with tf.GradientTape() as tape:
                logits = model(feats)
                loss = loss_fcn(labels, logits)
                for weight in model.trainable_weights:
                    loss = loss + \
                        args.weight_decay*tf.nn.l2_loss(weight)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if epoch >= 3:
                dur.append(time.time() - t0)

            loss_list.append(loss.numpy().item())
        if epoch >= 3:
            total_dur.append(np.sum(dur))
        loss_data = np.array(loss_list).mean()

        print("Epoch {:05d} | Time(s) {:.4f} | Loss: {:.4f}".format(epoch + 1, np.mean(total_dur), loss_data))
        if epoch % 5 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                score, val_loss = evaluate(feats, model, subgraph, labels, loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            print("Val F1-Score: {:.4f} ".format(mean_score))
            # early stop
            if mean_score > best_score or best_loss > mean_val_loss:
                if mean_score > best_score and best_loss > mean_val_loss:
                    val_early_loss = mean_val_loss
                    val_early_score = mean_score
                best_score = np.max((mean_score, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break
    test_score_list = []
    # for batch, test_data in enumerate(test_dataloader):
    #     subgraph, feats, labels = test_data
    #     feats = feats.to(device)
    #     labels = labels.to(device)
    #     test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    # print("Test F1-Score: {:.4f}".format(np.array(test_score_list).mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    print(args)

    main(args)
