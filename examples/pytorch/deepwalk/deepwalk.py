import torch
import argparse
import dgl
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import os
import random
import time
import numpy as np

from reading_data import DeepwalkDataset
from model import SkipGramModel
from utils import thread_wrapped_func, shuffle_walks

class DeepwalkTrainer:
    def __init__(self, args):
        """ Initializing the trainer with the input arguments """
        self.args = args
        self.dataset = DeepwalkDataset(
            net_file=args.data_file,
            map_file=args.map_file,
            walk_length=args.walk_length,
            window_size=args.window_size,
            num_walks=args.num_walks,
            batch_size=args.batch_size,
            negative=args.negative,
            gpus=args.gpus,
            fast_neg=args.fast_neg,
            )
        self.emb_size = len(self.dataset.net)
        self.emb_model = None

    def init_device_emb(self):
        """ set the device before training 
        will be called once in fast_train_mp / fast_train
        """
        choices = sum([self.args.only_gpu, self.args.only_cpu, self.args.mix])
        assert choices == 1, "Must choose only *one* training mode in [only_cpu, only_gpu, mix]"
        choices = sum([self.args.sgd, self.args.adam, self.args.avg_sgd])
        assert choices == 1, "Must choose only *one* gradient descent strategy in [sgd, avg_sgd, adam]"
        
        # initializing embedding on CPU
        self.emb_model = SkipGramModel(
            emb_size=self.emb_size, 
            emb_dimension=self.args.dim,
            walk_length=self.args.walk_length,
            window_size=self.args.window_size,
            batch_size=self.args.batch_size,
            only_cpu=self.args.only_cpu,
            only_gpu=self.args.only_gpu,
            mix=self.args.mix,
            neg_weight=self.args.neg_weight,
            negative=self.args.negative,
            lr=self.args.lr,
            lap_norm=self.args.lap_norm,
            adam=self.args.adam,
            sgd=self.args.sgd,
            avg_sgd=self.args.avg_sgd,
            fast_neg=self.args.fast_neg,
            )
        
        torch.set_num_threads(self.args.num_threads)
        if self.args.only_gpu:
            print("Run in 1 GPU")
            assert self.args.gpus[0] >= 0
            self.emb_model.all_to_device(self.args.gpus[0])
        elif self.args.mix:
            print("Mix CPU with %d GPU" % len(self.args.gpus))
            if len(self.args.gpus) == 1:
                assert self.args.gpus[0] >= 0, 'mix CPU with GPU should have abaliable GPU'
                self.emb_model.set_device(self.args.gpus[0])
        else:
            print("Run in CPU process")
            self.args.gpus = [torch.device('cpu')]


    def train(self):
        """ train the embedding """
        if len(self.args.gpus) > 1:
            self.fast_train_mp()
        else:
            self.fast_train()

    def fast_train_mp(self):
        """ multi-cpu-core or mix cpu & multi-gpu """
        self.init_device_emb()
        self.emb_model.share_memory()

        start_all = time.time()
        ps = []

        for i in range(len(self.args.gpus)):
            p = mp.Process(target=self.fast_train_sp, args=(self.args.gpus[i],))
            ps.append(p)
            p.start()

        for p in ps:
            p.join()
        
        print("Used time: %.2fs" % (time.time()-start_all))
        if self.args.save_in_txt:
            self.emb_model.save_embedding_txt(self.dataset, self.args.output_emb_file)
        else:
            self.emb_model.save_embedding(self.dataset, self.args.output_emb_file)

    @thread_wrapped_func
    def fast_train_sp(self, gpu_id):
        """ a subprocess for fast_train_mp """
        if self.args.mix:
            self.emb_model.set_device(gpu_id)
        torch.set_num_threads(self.args.num_threads)

        sampler = self.dataset.create_sampler(gpu_id)

        dataloader = DataLoader(
            dataset=sampler.seeds,
            batch_size=self.args.batch_size,
            collate_fn=sampler.sample,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            )
        num_batches = len(dataloader)
        print("num batchs: %d in subprocess [%d]" % (num_batches, gpu_id))
        # number of positive node pairs in a sequence
        num_pos = int(2 * self.args.walk_length * self.args.window_size\
            - self.args.window_size * (self.args.window_size + 1))
        
        start = time.time()
        with torch.no_grad():
            max_i = self.args.iterations * num_batches
            
            for i, walks in enumerate(dataloader):
                # decay learning rate for SGD
                lr = self.args.lr * (max_i - i) / max_i
                if lr < 0.00001:
                    lr = 0.00001

                if self.args.fast_neg:
                    self.emb_model.fast_learn(walks, lr)
                else:
                    # do negative sampling
                    bs = len(walks)
                    neg_nodes = torch.LongTensor(
                        np.random.choice(self.dataset.neg_table, 
                            bs * num_pos * self.args.negative, 
                            replace=True))
                    self.emb_model.fast_learn(walks, lr, neg_nodes=neg_nodes)

                if i > 0 and i % self.args.print_interval == 0:
                    print("Solver [%d] batch %d tt: %.2fs" % (gpu_id, i, time.time()-start))
                    start = time.time()

    def fast_train(self):
        """ fast train with dataloader """
        # the number of postive node pairs of a node sequence
        num_pos = 2 * self.args.walk_length * self.args.window_size\
            - self.args.window_size * (self.args.window_size + 1)
        num_pos = int(num_pos)

        self.init_device_emb()

        sampler = self.dataset.create_sampler(0)

        dataloader = DataLoader(
            dataset=sampler.seeds,
            batch_size=self.args.batch_size,
            collate_fn=sampler.sample,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            )
        
        num_batches = len(dataloader)
        print("num batchs: %d" % num_batches)

        start_all = time.time()
        start = time.time()
        with torch.no_grad():
            max_i = self.args.iterations * num_batches
            for iteration in range(self.args.iterations):
                print("\nIteration: " + str(iteration + 1))
                
                for i, walks in enumerate(dataloader):
                    # decay learning rate for SGD
                    lr = self.args.lr * (max_i - i) / max_i
                    if lr < 0.00001:
                        lr = 0.00001

                    if self.args.fast_neg:
                        self.emb_model.fast_learn(walks, lr)
                    else:
                        # do negative sampling
                        bs = len(walks)
                        neg_nodes = torch.LongTensor(
                            np.random.choice(self.dataset.neg_table, 
                                bs * num_pos * self.args.negative, 
                                replace=True))
                        self.emb_model.fast_learn(walks, lr, neg_nodes=neg_nodes)

                    if i > 0 and i % self.args.print_interval == 0:
                        print("Batch %d, training time: %.2fs" % (i, time.time()-start))
                        start = time.time()

        print("Training used time: %.2fs" % (time.time()-start_all))
        if self.args.save_in_txt:
            self.emb_model.save_embedding_txt(self.dataset, self.args.output_emb_file)
        else:
            self.emb_model.save_embedding(self.dataset, self.args.output_emb_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepWalk")
    parser.add_argument('--data_file', type=str, 
            help="path of the txt network file, builtin dataset include youtube-net and blog-net") 
    parser.add_argument('--save_in_txt', default=False, action="store_true",
            help='Whether save dat in txt format or npy')
    parser.add_argument('--output_emb_file', type=str, default="emb.npy",
            help='path of the output npy embedding file')
    parser.add_argument('--map_file', type=str, default="nodeid_to_index.pickle",
            help='path of the mapping dict that maps node ids to embedding index')
    parser.add_argument('--dim', default=128, type=int, 
            help="embedding dimensions")
    parser.add_argument('--window_size', default=5, type=int, 
            help="context window size")
    parser.add_argument('--num_walks', default=10, type=int, 
            help="number of walks for each node")
    parser.add_argument('--negative', default=5, type=int, 
            help="negative samples for each positve node pair")
    parser.add_argument('--iterations', default=1, type=int, 
            help="iterations")
    parser.add_argument('--batch_size', default=10, type=int, 
            help="number of node sequences in each batch")
    parser.add_argument('--print_interval', default=1000, type=int, 
            help="number of batches between printing")
    parser.add_argument('--walk_length', default=80, type=int, 
            help="number of nodes in a sequence")
    parser.add_argument('--lr', default=0.2, type=float, 
            help="learning rate")
    parser.add_argument('--neg_weight', default=1., type=float, 
            help="negative weight")
    parser.add_argument('--lap_norm', default=0.01, type=float, 
            help="weight of laplacian normalization")
    parser.add_argument('--mix', default=False, action="store_true", 
            help="mixed training with CPU and GPU")
    parser.add_argument('--only_cpu', default=False, action="store_true", 
            help="training with CPU")
    parser.add_argument('--only_gpu', default=False, action="store_true", 
            help="training with GPU")
    parser.add_argument('--fast_neg', default=True, action="store_true", 
            help="do negative sampling inside a batch")
    parser.add_argument('--adam', default=False, action="store_true", 
            help="use adam for embedding updation, recommended")
    parser.add_argument('--sgd', default=False, action="store_true", 
            help="use sgd for embedding updation")
    parser.add_argument('--avg_sgd', default=False, action="store_true", 
            help="average gradients of sgd for embedding updation")
    parser.add_argument('--num_threads', default=2, type=int, 
            help="number of threads used for each CPU-core/GPU")
    parser.add_argument('--gpus', type=int, default=[-1], nargs='+', 
            help='a list of active gpu ids, e.g. 0')
    args = parser.parse_args()

    start_time = time.time()
    trainer = DeepwalkTrainer(args)
    trainer.train()
    print("Total used time: %.2f" % (time.time() - start_time))
