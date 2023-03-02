import argparse

import torch
import torch.optim as optim
from download import AminerDataset, CustomDataset
from model import SkipGramModel

from reading_data import DataReader, Metapath2vecDataset
from torch.utils.data import DataLoader

from tqdm import tqdm


class Metapath2VecTrainer:
    def __init__(self, args):
        if args.aminer:
            dataset = AminerDataset(args.path)
        else:
            dataset = CustomDataset(args.path)
        self.data = DataReader(dataset, args.min_count, args.care_type)
        dataset = Metapath2vecDataset(self.data, args.window_size)
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=dataset.collate,
        )

        self.output_file_name = args.output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        optimizer = optim.SparseAdam(
            list(self.skip_gram_model.parameters()), lr=self.initial_lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(self.dataloader)
        )

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metapath2vec")
    # parser.add_argument('--input_file', type=str, help="input_file")
    parser.add_argument(
        "--aminer", action="store_true", help="Use AMiner dataset"
    )
    parser.add_argument("--path", type=str, help="input_path")
    parser.add_argument("--output_file", type=str, help="output_file")
    parser.add_argument(
        "--dim", default=128, type=int, help="embedding dimensions"
    )
    parser.add_argument(
        "--window_size", default=7, type=int, help="context window size"
    )
    parser.add_argument("--iterations", default=5, type=int, help="iterations")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")
    parser.add_argument(
        "--care_type",
        default=0,
        type=int,
        help="if 1, heterogeneous negative sampling, else normal negative sampling",
    )
    parser.add_argument(
        "--initial_lr", default=0.025, type=float, help="learning rate"
    )
    parser.add_argument("--min_count", default=5, type=int, help="min count")
    parser.add_argument(
        "--num_workers", default=16, type=int, help="number of workers"
    )
    args = parser.parse_args()
    m2v = Metapath2VecTrainer(args)
    m2v.train()
