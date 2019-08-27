import torch
from torch.utils.data import DataLoader

import argparse
from dgl import model_zoo
import rdkit

from jtnn import *


def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


worker_init_fn(None)

parser = argparse.ArgumentParser(description="Evaluation for JTNN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--train", dest="train",
                    default='test', help='Training file name')
parser.add_argument("-v", "--vocab", dest="vocab",
                    default='vocab', help='Vocab file name')
parser.add_argument("-m", "--model", dest="model_path", default=None,
                    help="Pre-trained model to be loaded for evalutaion. If not specified,"
                         " would use pre-trained model from model zoo")
parser.add_argument("-w", "--hidden", dest="hidden_size", default=450,
                    help="Hidden size of representation vector, "
                         "should be consistent with pre-trained model")
parser.add_argument("-l", "--latent", dest="latent_size", default=56,
                    help="Latent Size of node(atom) features and edge(atom) features, "
                         "should be consistent with pre-trained model")
parser.add_argument("-d", "--depth", dest="depth", default=3,
                    help="Depth of message passing hops, "
                         "should be consistent with pre-trained model")
args = parser.parse_args()

dataset = JTNNDataset(data=args.train, vocab=args.vocab, training=False)
vocab_file = dataset.vocab_file

hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)

model = model_zoo.chem.DGLJTNNVAE(vocab_file=vocab_file,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size,
                                  depth=depth)

if args.model_path is not None:
    model.load_state_dict(torch.load(args.model_path))
else:
    model = model_zoo.chem.load_pretrained("JTNN_ZINC")

model = cuda(model)
model.eval()
print("Model #Params: %dK" %
      (sum([x.nelement() for x in model.parameters()]) / 1000,))

MAX_EPOCH = 100
PRINT_ITER = 20


def reconstruct():
    dataset.training = False
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=JTNNCollator(dataset.vocab, False),
        drop_last=True,
        worker_init_fn=worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    acc = 0.0
    tot = 0
    with torch.no_grad():
        for it, batch in enumerate(dataloader):
            gt_smiles = batch['mol_trees'][0].smiles
            # print(gt_smiles)
            model.move_to_cuda(batch)
            try:
                _, tree_vec, mol_vec = model.encode(batch)

                tree_mean = model.T_mean(tree_vec)
                # Following Mueller et al.
                tree_log_var = -torch.abs(model.T_var(tree_vec))
                mol_mean = model.G_mean(mol_vec)
                # Following Mueller et al.
                mol_log_var = -torch.abs(model.G_var(mol_vec))

                epsilon = torch.randn(1, model.latent_size // 2).cuda()
                tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
                epsilon = torch.randn(1, model.latent_size // 2).cuda()
                mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
                dec_smiles = model.decode(tree_vec, mol_vec)

                if dec_smiles == gt_smiles:
                    acc += 1
                tot += 1
            except Exception as e:
                print("Failed to encode: {}".format(gt_smiles))
                print(e)

            if it % 20 == 1:
                print("Progress {}/{}; Current Reconstruction Accuracy: {:.4f}".format(it,
                                                                                       len(dataloader), acc / tot))
    return acc / tot


if __name__ == '__main__':
    reconstruct_acc = reconstruct()
    print("Reconstruction Accuracy: {}".format(reconstruct_acc))
