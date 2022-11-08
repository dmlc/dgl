import random

import numpy as np
import torch
import torch.nn.functional as F
from models import MWE_DGCN, MWE_GCN


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("random seed set to be " + str(seed))


def load_model(args):
    if args["model"] == "MWE-GCN":
        model = MWE_GCN(
            n_input=args["in_feats"],
            n_hidden=args["hidden_feats"],
            n_output=args["out_feats"],
            n_layers=args["n_layers"],
            activation=torch.nn.Tanh(),
            dropout=args["dropout"],
            aggr_mode=args["aggr_mode"],
            device=args["device"],
        )
    elif args["model"] == "MWE-DGCN":
        model = MWE_DGCN(
            n_input=args["in_feats"],
            n_hidden=args["hidden_feats"],
            n_output=args["out_feats"],
            n_layers=args["n_layers"],
            activation=torch.nn.ReLU(),
            dropout=args["dropout"],
            aggr_mode=args["aggr_mode"],
            residual=args["residual"],
            device=args["device"],
        )
    else:
        raise ValueError("Unexpected model {}".format(args["model"]))

    return model


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")
