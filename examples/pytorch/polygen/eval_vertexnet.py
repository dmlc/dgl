# Beam Search Module

from modules import *
from dataset import *
from tqdm import tqdm
import numpy as n
import argparse

k = 5 # Beam size

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('testing translation model')
    argparser.add_argument('--gpu', default=-1, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='num of layers')
    argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=64, help='batch size')
    argparser.add_argument('--ckpt-dir', default='.', type=str, help='checkpoint path')
    argparser.add_argument('--epoch', default=1, help='epoch number')
    args = argparser.parse_args()
    args_filter = ['batch', 'gpu']
    exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
    device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)

    dataset = get_dataset('vertex')
    V = dataset.vocab_size
    dim_model = 256

    graph_pool = VertexNetGraphPool()
    model = make_vertex_model(N=args.N, dim_model=dim_model)
    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.'+str(args.epoch)+'.pt')
    with open(ckpt_path, 'rb') as f:
        model.load_state_dict(th.load(f, map_location=lambda storage, loc: storage))
    model = model.to(device)
    model.eval()
    test_iter = dataset(graph_pool, mode='test', batch_size=args.batch, devices=[device], k=k)
    for i, g in enumerate(test_iter):
        with th.no_grad():
            output = model.infer(g, dataset.MAX_LENGTH, dataset.eos_id, k, alpha=0.6)
        for line in dataset.get_sequence(output):
            print(line, file=fpred)
        for line in dataset.tgt['test']:
            print(line.strip(), file=fref)
    fpred.close()
    fref.close()
