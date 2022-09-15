import sys
from time import strftime, localtime
import logging
import torch
import argparse
import torch.optim as optim
from dgl.nn.pytorch import MetaPath2Vec
from tqdm import tqdm
from reading_data import MetaPathGenerator


class MetaPath2VecTrainer:
    def __init__(self, args):
        mpg = MetaPathGenerator()
        mpg.read_data(args.input_path)
        hg, id_conf, id_author = mpg.generate_random_aca()
        self.data =hg
        nid2word = {"conference": id_conf,
                    "author": id_author}
        self.metapath=args.meta_path
        self.node_repeat=args.node_repeat
        self.window_size=args.window_size
        self.min_count=args.min_count
        self.negative_samples=args.negative_samples
        self.output_file_name = args.output_file
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.num_workers=args.num_workers
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.model = MetaPath2Vec(self.data,self.emb_dimension,self.metapath,self.window_size,self.min_count,
                                  self.negative_samples,self.node_repeat,nid2word)

        self.dataloader= self.model.loader(batch_size=self.batch_size, num_workers=self.num_workers)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        log_file = '{}-{}-{}.log'.format("dgl_meta", "aminer", strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))

        self.model.initParameters()
        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for iteration in range(self.iterations):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(iteration))
            print("\n\n\nIteration: " + str(iteration + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                pos_u = sample_batched[0].to(self.device)
                pos_v = sample_batched[1].to(self.device)
                neg_v = sample_batched[2].to(self.device)

                scheduler.step()
                optimizer.zero_grad()
                loss = self.model.forward(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()

                running_loss = running_loss * 0.9 + loss.item() * 0.1
                if i > 0 and i % 500 == 0:
                    now = localtime()
                    now_time = strftime("%Y-%m-%d %H:%M:%S", now)
                    logger.info('time:{} loss: {:.4f}'.format(now_time, running_loss))
                    print(" Loss: " + str(running_loss))

            save_embedding(self.model,self.model.id2word,"out",iteration)


def save_embedding(model,id2word, file_name, num):
    embedding = model.u_embeddings.weight.cpu().data.numpy()
    with open(file_name + "/" + file_name + str(num) +strftime("%y%m%d-%H%M", localtime()) +".txt", 'w') as f:
        f.write('%d %d\n' % (len(id2word), model.emb_dimension))
        for wid, w in id2word.items():
            e = ' '.join(map(lambda x: str(x), embedding[wid]))
            f.write('%s %s\n' % (w, e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaPath2Vec")
    parser.add_argument('--input_path', type=str, help="input_path")
    parser.add_argument('--meta_path', default=['ca', 'ac'] * 100,type=list, help="metapth composed of edge type sequence")
    parser.add_argument('--node_repeat', default=1000,type=int, help="The number of random walks to sample for each start node")
    parser.add_argument('--output_file', type=str, help='output_file')
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    parser.add_argument('--iterations', default=5, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--negative_samples', default=5, type=int, help="The number of negative samples need sampling")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")
    args = parser.parse_args()
    m2v = MetaPath2VecTrainer(args)
    m2v.train()
