"""
Differences compared to tkipf/relation-gcn
* weight decay applied to all weights
"""
import argparse
import gc
import torch as th
import torch.nn.functional as F
import dgl
import torch.multiprocessing as mp

from torchmetrics.functional import accuracy
from torch.nn.parallel import DistributedDataParallel

from entity_utils import load_data
from entity_sample import init_dataloaders, train, evaluate
from model import RGCN

def collect_eval(n_gpus, queue, labels):
    eval_logits = []
    eval_seeds = []
    for _ in range(n_gpus):
        eval_l, eval_s = queue.get()
        eval_logits.append(eval_l)
        eval_seeds.append(eval_s)
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    eval_acc = accuracy(eval_logits.argmax(dim=1), labels[eval_seeds].cpu()).item()

    return eval_acc

def run(proc_id, n_gpus, n_cpus, args, devices, dataset, queue=None):
    dev_id = devices[proc_id]
    th.cuda.set_device(dev_id)
    g, num_rels, num_classes, labels, train_idx, test_idx,\
        target_idx, inv_target = dataset

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    backend = 'nccl'
    if proc_id == 0:
        print("backend using {}".format(backend))
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=n_gpus,
                                      rank=proc_id)

    device = th.device(dev_id)
    use_ddp = True if n_gpus > 1 else False
    train_loader, val_loader, test_loader = init_dataloaders(
        args, g, train_idx, test_idx, target_idx, dev_id, use_ddp=use_ddp)

    model = RGCN(g.num_nodes(),
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases,
                 dropout=args.dropout,
                 self_loop=args.use_self_loop,
                 ns_mode=True)
    labels = labels.to(device)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=args.wd)

    th.set_num_threads(n_cpus)
    for epoch in range(args.n_epochs):
        train_acc, loss = train(model, train_loader, inv_target,
                                labels, optimizer)

        if proc_id == 0:
            print("Epoch {:05d}/{:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
                epoch, args.n_epochs, train_acc, loss))

        # garbage collection that empties the queue
        gc.collect()

        val_logits, val_seeds = evaluate(model, val_loader, inv_target)
        queue.put((val_logits, val_seeds))

        # gather evaluation result from multiple processes
        if proc_id == 0:
            val_acc = collect_eval(n_gpus, queue, labels)
            print("Validation Accuracy: {:.4f}".format(val_acc))

    # garbage collection that empties the queue
    gc.collect()
    test_logits, test_seeds = evaluate(model, test_loader, inv_target)
    queue.put((test_logits, test_seeds))
    if proc_id == 0:
        test_acc = collect_eval(n_gpus, queue, labels)
        print("Final Test Accuracy: {:.4f}".format(test_acc))
    th.distributed.barrier()

def main(args, devices):
    data = load_data(args.dataset, inv_target=True)

    # Create csr/coo/csc formats before launching training processes.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g = data[0]
    g.create_formats_()

    n_gpus = len(devices)
    # required for mp.Queue() to work with mp.spawn()
    mp.set_start_method('spawn')
    n_cpus = mp.cpu_count()
    queue = mp.Queue(n_gpus)
    mp.spawn(run, args=(n_gpus, n_cpus // n_gpus, args, devices, data, queue),
             nprocs=n_gpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification with sampling and multiple gpus')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--fanout", type=str, default="4, 4",
                        help="Fan-out of neighbor sampling")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Mini-batch size. ")
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(',')))

    print(args)
    main(args, devices)
