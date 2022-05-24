import argparse
from itertools import chain
from timeit import default_timer
from typing import Callable, Tuple, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from model import EntityClassify, RelGraphEmbedding


def train(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    embedding_optimizer: torch.optim.Optimizer,
    model_optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    predict_category: str,
    dataloader: dgl.dataloading.DataLoader,
) -> Tuple[float]:
    model.train()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    embedding_layer = embedding_layer.to(device)
    model = model.to(device)
    loss_function = loss_function.to(device)

    for step, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
        embedding_optimizer.zero_grad()
        model_optimizer.zero_grad()

        in_nodes = {rel: nid.to(device) for rel, nid in in_nodes.items()}
        out_nodes = out_nodes[predict_category].to(device)
        blocks = [block.to(device) for block in blocks]

        batch_labels = labels[out_nodes].to(device)

        embedding = embedding_layer(in_nodes=in_nodes, device=device)
        logits = model(blocks, embedding)[predict_category]

        loss = loss_function(logits, batch_labels)

        indices = logits.argmax(dim=-1)
        correct = torch.sum(indices == batch_labels)
        accuracy = correct.item() / len(batch_labels)

        loss.backward()
        model_optimizer.step()
        embedding_optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_accuracy /= step + 1

    return time, total_loss, total_accuracy


def validate(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    inference_mode: str,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hg: dgl.DGLHeteroGraph,
    labels: torch.Tensor,
    predict_category: str,
    dataloader: dgl.dataloading.DataLoader = None,
    eval_batch_size: int = None,
    eval_num_workers: int = None,
    mask: torch.Tensor = None,
) -> Tuple[float]:
    embedding_layer.eval()
    model.eval()

    start = default_timer()

    embedding_layer = embedding_layer.to(device)
    model = model.to(device)
    loss_function = loss_function.to(device)

    valid_labels = labels[mask].to(device)

    with torch.no_grad():
        if inference_mode == 'neighbor_sampler':
            total_loss = 0
            total_accuracy = 0

            for step, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
                in_nodes = {rel: nid.to(device)
                            for rel, nid in in_nodes.items()}
                out_nodes = out_nodes[predict_category].to(device)
                blocks = [block.to(device) for block in blocks]

                batch_labels = labels[out_nodes].to(device)

                embedding = embedding_layer(in_nodes=in_nodes, device=device)
                logits = model(blocks, embedding)[predict_category]

                loss = loss_function(logits, batch_labels)

                indices = logits.argmax(dim=-1)
                correct = torch.sum(indices == batch_labels)
                accuracy = correct.item() / len(batch_labels)

                total_loss += loss.item()
                total_accuracy += accuracy

            total_loss /= step + 1
            total_accuracy /= step + 1
        elif inference_mode == 'full_neighbor_sampler':
            logits = model.inference(
                hg,
                eval_batch_size,
                eval_num_workers,
                embedding_layer,
                device,
            )[predict_category][mask]

            total_loss = loss_function(logits, valid_labels)

            indices = logits.argmax(dim=-1)
            correct = torch.sum(indices == valid_labels)
            total_accuracy = correct.item() / len(valid_labels)

            total_loss = total_loss.item()
        else:
            embedding = embedding_layer(device=device)
            logits = model(hg, embedding)[predict_category][mask]

            total_loss = loss_function(logits, valid_labels)

            indices = logits.argmax(dim=-1)
            correct = torch.sum(indices == valid_labels)
            total_accuracy = correct.item() / len(valid_labels)

            total_loss = total_loss.item()

    stop = default_timer()
    time = stop - start

    return time, total_loss, total_accuracy


def run(args: argparse.ArgumentParser) -> None:
    torch.manual_seed(args.seed)

    dataset, hg, train_idx, valid_idx, test_idx = utils.process_dataset(
        args.dataset,
        root=args.dataset_root,
    )
    predict_category = dataset.predict_category
    labels = hg.nodes[predict_category].data['labels']

    training_device = torch.device('cuda' if args.gpu_training else 'cpu')
    inference_device = torch.device('cuda' if args.gpu_inference else 'cpu')

    inferfence_mode = args.inference_mode

    fanouts = [int(fanout) for fanout in args.fanouts.split(',')]

    train_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    train_dataloader = dgl.dataloading.DataLoader(
        hg,
        {predict_category: train_idx},
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if inferfence_mode == 'neighbor_sampler':
        valid_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        valid_dataloader = dgl.dataloading.DataLoader(
            hg,
            {predict_category: valid_idx},
            valid_sampler,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.eval_num_workers,
        )

        if args.test_validation:
            test_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
            test_dataloader = dgl.dataloading.DataLoader(
                hg,
                {predict_category: test_idx},
                test_sampler,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.eval_num_workers,
            )
    else:
        valid_dataloader = None

        if args.test_validation:
            test_dataloader = None

    in_feats = hg.nodes[predict_category].data['feat'].shape[-1]
    out_feats = dataset.num_classes

    num_nodes = {}
    node_feats = {}

    for ntype in hg.ntypes:
        num_nodes[ntype] = hg.num_nodes(ntype)
        node_feats[ntype] = hg.nodes[ntype].data.get('feat')

    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    embedding_layer = RelGraphEmbedding(hg, in_feats, num_nodes, node_feats)
    model = EntityClassify(
        hg,
        in_feats,
        args.hidden_feats,
        out_feats,
        args.num_bases,
        args.num_layers,
        norm=args.norm,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        dropout=args.dropout,
        activation=activations[args.activation],
        self_loop=args.self_loop,
    )

    loss_function = nn.CrossEntropyLoss()

    embedding_optimizer = torch.optim.SparseAdam(
        embedding_layer.node_embeddings.parameters(), lr=args.embedding_lr)

    if args.node_feats_projection:
        all_parameters = chain(
            model.parameters(), embedding_layer.embeddings.parameters())
        model_optimizer = torch.optim.Adam(all_parameters, lr=args.model_lr)
    else:
        model_optimizer = torch.optim.Adam(
            model.parameters(), lr=args.model_lr)

    checkpoint = utils.Callback(args.early_stopping_patience,
                                args.early_stopping_monitor)

    print('## Training started ##')

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_accuracy = train(
            embedding_layer,
            model,
            training_device,
            embedding_optimizer,
            model_optimizer,
            loss_function,
            labels,
            predict_category,
            train_dataloader,
        )
        valid_time, valid_loss, valid_accuracy = validate(
            embedding_layer,
            model,
            inference_device,
            inferfence_mode,
            loss_function,
            hg,
            labels,
            predict_category=predict_category,
            dataloader=valid_dataloader,
            eval_batch_size=args.eval_batch_size,
            eval_num_workers=args.eval_num_workers,
            mask=valid_idx,
        )

        checkpoint.create(
            epoch,
            train_time,
            valid_time,
            train_loss,
            valid_loss,
            train_accuracy,
            valid_accuracy,
            {'embedding_layer': embedding_layer, 'model': model},
        )

        print(
            f'Epoch: {epoch + 1:03} '
            f'Train Loss: {train_loss:.2f} '
            f'Valid Loss: {valid_loss:.2f} '
            f'Train Accuracy: {train_accuracy:.4f} '
            f'Valid Accuracy: {valid_accuracy:.4f} '
            f'Train Epoch Time: {train_time:.2f} '
            f'Valid Epoch Time: {valid_time:.2f}'
        )

        if checkpoint.should_stop:
            print('## Training finished: early stopping ##')

            break
        elif epoch >= args.num_epochs - 1:
            print('## Training finished ##')

    print(
        f'Best Epoch: {checkpoint.best_epoch} '
        f'Train Loss: {checkpoint.best_epoch_train_loss:.2f} '
        f'Valid Loss: {checkpoint.best_epoch_valid_loss:.2f} '
        f'Train Accuracy: {checkpoint.best_epoch_train_accuracy:.4f} '
        f'Valid Accuracy: {checkpoint.best_epoch_valid_accuracy:.4f}'
    )

    if args.test_validation:
        print('## Test data validation ##')

        embedding_layer.load_state_dict(
            checkpoint.best_epoch_model_parameters['embedding_layer'])
        model.load_state_dict(checkpoint.best_epoch_model_parameters['model'])

        test_time, test_loss, test_accuracy = validate(
            embedding_layer,
            model,
            inference_device,
            inferfence_mode,
            loss_function,
            hg,
            labels,
            predict_category=predict_category,
            dataloader=test_dataloader,
            eval_batch_size=args.eval_batch_size,
            eval_num_workers=args.eval_num_workers,
            mask=test_idx,
        )

        print(
            f'Test Loss: {test_loss:.2f} '
            f'Test Accuracy: {test_accuracy:.4f} '
            f'Test Epoch Time: {test_time:.2f}'
        )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('RGCN')

    argparser.add_argument('--gpu-training', dest='gpu_training',
                           action='store_true')
    argparser.add_argument('--no-gpu-training', dest='gpu_training',
                           action='store_false')
    argparser.set_defaults(gpu_training=True)
    argparser.add_argument('--gpu-inference', dest='gpu_inference',
                           action='store_true')
    argparser.add_argument('--no-gpu-inference', dest='gpu_inference',
                           action='store_false')
    argparser.set_defaults(gpu_inference=True)
    argparser.add_argument('--inference-mode', default='neighbor_sampler', type=str,
                           choices=['neighbor_sampler', 'full_neighbor_sampler', 'full_graph'])
    argparser.add_argument('--dataset', default='ogbn-mag', type=str,
                           choices=['ogbn-mag'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--embedding-lr', default=0.01, type=float)
    argparser.add_argument('--model-lr', default=0.01, type=float)
    argparser.add_argument('--node-feats-projection',
                           dest='node_feats_projection', action='store_true')
    argparser.add_argument('--no-node-feats-projection',
                           dest='node_feats_projection', action='store_false')
    argparser.set_defaults(node_feats_projection=False)
    argparser.add_argument('--hidden-feats', default=64, type=int)
    argparser.add_argument('--num-bases', default=2, type=int)
    argparser.add_argument('--num-layers', default=2, type=int)
    argparser.add_argument('--norm', default='right',
                           type=str, choices=['both', 'none', 'right'])
    argparser.add_argument('--layer-norm', dest='layer_norm',
                           action='store_true')
    argparser.add_argument('--no-layer-norm', dest='layer_norm',
                           action='store_false')
    argparser.set_defaults(layer_norm=False)
    argparser.add_argument('--input-dropout', default=0.1, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--activation', default='relu', type=str,
                           choices=['leaky_relu', 'relu'])
    argparser.add_argument('--self-loop', dest='self_loop',
                           action='store_true')
    argparser.add_argument('--no-self-loop', dest='self_loop',
                           action='store_false')
    argparser.set_defaults(self_loop=True)
    argparser.add_argument('--fanouts', default='25,20', type=str)
    argparser.add_argument('--batch-size', default=1024, type=int)
    argparser.add_argument('--eval-batch-size', default=1024, type=int)
    argparser.add_argument('--num-workers', default=4, type=int)
    argparser.add_argument('--eval-num-workers', default=4, type=int)
    argparser.add_argument('--early-stopping-patience', default=10, type=int)
    argparser.add_argument('--early-stopping-monitor', default='loss',
                           type=str, choices=['accuracy', 'loss'])
    argparser.add_argument('--test-validation', dest='test_validation',
                           action='store_true')
    argparser.add_argument('--no-test-validation', dest='test_validation',
                           action='store_false')
    argparser.set_defaults(test_validation=True)
    argparser.add_argument('--seed', default=13, type=int)

    args = argparser.parse_args()

    run(args)
