"""
Differences compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import gc
import torch as th
import torch.nn.functional as F
import dgl.multiprocessing as mp
import dgl

from torch.nn.parallel import DistributedDataParallel

from entity_utils import load_data
from entity_sample import gen_norm, init_dataloaders, init_models, train, evaluate

def run(proc_id, n_gpus, n_cpus, args, devices, dataset, queue=None):
    dev_id = devices[proc_id]
    g, node_feats, num_classes, num_rels, target_idx, inv_target, train_idx,\
        test_idx, labels = dataset

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    backend = 'nccl'

    # using sparse embedding or using mix_cpu_gpu model (embedding model can not be stored on GPU)
    if not args.dgl_sparse:
        backend = 'gloo'
    if proc_id == 0:
        print("backend using {}".format(backend))
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=n_gpus,
                                      rank=proc_id)

    device = th.device(dev_id)
    train_loader, val_loader, test_loader = init_dataloaders(
        args, g, train_idx, test_idx, target_idx, dev_id, n_gpus)
    embed_layer, model = init_models(args, device, node_feats, num_classes, num_rels)

    if n_gpus > 1:
        labels = labels.to(device)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if args.dgl_sparse:
            embed_layer.cuda(dev_id)
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        else:
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)

    # optimizer
    if args.dgl_sparse:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=1e-2, weight_decay=args.l2norm)
        emb_optimizer = dgl.optim.SparseAdam(params=embed_layer.module.dgl_emb,
                                             lr=args.sparse_lr, eps=1e-8)
    else:
        dense_params = list(model.parameters())
        optimizer = th.optim.Adam(dense_params, lr=1e-2, weight_decay=args.l2norm)
        embs = list(embed_layer.module.node_embeds.parameters())
        emb_optimizer = th.optim.SparseAdam(embs, lr=args.sparse_lr)

    th.set_num_threads(n_cpus)
    for epoch in range(args.n_epochs):
        train_loader.set_epoch(epoch)
        train_acc, loss = train(model, embed_layer, train_loader, inv_target, device,
                                labels, emb_optimizer, optimizer)

        if proc_id == 0:
            print("Epoch {:05d}/{:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
                epoch, args.n_epochs, train_acc, loss.item()))

        gc.collect()

        def collect_eval():
            eval_logits = []
            eval_seeds = []
            for _ in range(n_gpus):
                eval_l, eval_s = queue.get()
                eval_logits.append(eval_l)
                eval_seeds.append(eval_s)
            eval_logits = th.cat(eval_logits)
            eval_seeds = th.cat(eval_seeds)
            eval_loss = F.cross_entropy(eval_logits, labels[eval_seeds].cpu()).item()
            eval_acc = th.mean(eval_logits.argmax(dim=1) == labels[eval_seeds].cpu()).item()

            return eval_loss, eval_acc

        val_logits, val_seeds = evaluate(device, model, embed_layer, val_loader, inv_target)
        queue.put((val_logits, val_seeds))

        # gather evaluation result from multiple processes
        if proc_id == 0:
            val_loss, val_acc = collect_eval()
            print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
                val_acc, val_loss))

    test_logits, test_seeds = evaluate(device, model, embed_layer, test_loader, inv_target)
    queue.put((test_logits, test_seeds))
    if proc_id == 0:
        test_loss, test_acc = collect_eval()
        print("Final Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))

def main(args, devices):
    hg, g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target = load_data(
        args.dataset, inv_target=True)
    node_feats = [hg.num_nodes(ntype) for ntype in hg.ntypes]

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.ndata[dgl.NID].share_memory_()
    target_idx.share_memory_()
    train_idx.share_memory_()
    test_idx.share_memory_()
    inv_target.share_memory_()

    # Create csr/coo/csc formats before launching training processes.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    n_gpus = len(devices)
    n_cpus = mp.cpu_count()
    queue = mp.Queue(n_gpus)
    procs = []
    for proc_id in range(n_gpus):
        # We use distributed data parallel dataloader to handle the data splitting
        p = mp.Process(target=run, args=(proc_id, n_gpus, n_cpus // n_gpus, args, devices,
                                        (g, node_feats, num_classes, num_rels, target_idx,
                                         inv_target, train_idx, test_idx, labels),
                                         queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification with sampling and multiple gpus')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu")
    parser.add_argument("--sparse-lr", type=float, default=2e-2,
                        help="sparse embedding learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=5e-4,
                        help="l2 norm coef")
    parser.add_argument("--fanout", type=str, default="4, 4",
                        help="Fan-out of neighbor sampling")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Mini-batch size. ")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
                        help='Use sparse embedding for node embeddings.')
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(',')))

    print(args)
    main(args, devices)
