# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/parser.py>.

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument(
        "--weights_path", nargs="?", default="model/", help="Store model path."
    )
    parser.add_argument(
        "--data_path", nargs="?", default="../Data/", help="Input data path."
    )
    parser.add_argument(
        "--model_name", type=str, default="NGCF.pkl", help="Saved model name."
    )

    parser.add_argument(
        "--dataset",
        nargs="?",
        default="gowalla",
        help="Choose a dataset from {gowalla, yelp2018, amazon-book}",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Interval of evaluation."
    )
    parser.add_argument(
        "--epoch", type=int, default=400, help="Number of epoch."
    )

    parser.add_argument(
        "--embed_size", type=int, default=64, help="Embedding size."
    )
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[64,64,64]",
        help="Output sizes of every layer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size."
    )

    parser.add_argument(
        "--regs", nargs="?", default="[1e-5]", help="Regularizations."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate."
    )

    parser.add_argument(
        "--gpu", type=int, default=0, help="0 for NAIS_prod, 1 for NAIS_concat"
    )

    parser.add_argument(
        "--mess_dropout",
        nargs="?",
        default="[0.1,0.1,0.1]",
        help="Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.",
    )

    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[20, 40]",
        help="Output sizes of every layer",
    )

    parser.add_argument(
        "--save_flag",
        type=int,
        default=1,
        help="0: Disable model saver, 1: Activate model saver",
    )

    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    parser.add_argument(
        "--report",
        type=int,
        default=0,
        help="0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels",
    )
    return parser.parse_args()
