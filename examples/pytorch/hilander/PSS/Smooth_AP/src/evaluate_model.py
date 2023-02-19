import argparse
import os

import auxiliaries as aux
import datasets as data
import evaluate as eval
import netlib as netlib
import torch

if __name__ == "__main__":
    ################## INPUT ARGUMENTS ###################
    parser = argparse.ArgumentParser()
    ####### Main Parameter: Dataset to use for Training
    parser.add_argument(
        "--dataset",
        default="vehicle_id",
        type=str,
        help="Dataset to use.",
        choices=["Inaturalist", "vehicle_id"],
    )
    parser.add_argument(
        "--source_path",
        default="/scratch/shared/beegfs/abrown/datasets",
        type=str,
        help="Path to training data.",
    )
    parser.add_argument(
        "--save_path",
        default=os.getcwd() + "/Training_Results",
        type=str,
        help="Where to save everything.",
    )
    parser.add_argument(
        "--savename",
        default="",
        type=str,
        help="Save folder name if any special information is to be included.",
    )

    ### General Training Parameters
    parser.add_argument(
        "--kernels",
        default=8,
        type=int,
        help="Number of workers for pytorch dataloader.",
    )
    parser.add_argument(
        "--bs", default=112, type=int, help="Mini-Batchsize to use."
    )
    parser.add_argument(
        "--samples_per_class",
        default=4,
        type=int,
        help="Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.",
    )
    parser.add_argument("--loss", default="smoothap", type=str)

    ##### Evaluation Settings
    parser.add_argument(
        "--k_vals",
        nargs="+",
        default=[1, 2, 4, 8],
        type=int,
        help="Recall @ Values.",
    )
    ##### Network parameters
    parser.add_argument(
        "--embed_dim",
        default=512,
        type=int,
        help="Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.",
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        help="Network backend choice: resnet50, googlenet, BNinception",
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU-id for GPU to use."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to where weights to be evaluated are saved.",
    )
    parser.add_argument(
        "--not_pretrained",
        action="store_true",
        help="If added, the network will be trained WITHOUT ImageNet-pretrained weights.",
    )

    parser.add_argument("--trainset", default="lin_train_set1.txt", type=str)
    parser.add_argument(
        "--testset", default="Inaturalist_test_set1.txt", type=str
    )
    parser.add_argument("--cluster_path", default="", type=str)
    parser.add_argument("--finetune", default="false", type=str)
    parser.add_argument("--class_num", default=948, type=int)
    parser.add_argument("--get_features", default="false", type=str)
    parser.add_argument(
        "--patch_size", default=16, type=int, help="vit patch size"
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="pretrained weight path",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=aux.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--drop_path_rate",
        default=0.1,
        type=float,
        help="stochastic depth rate",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=aux.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--linsize", default=29011, type=int, help="Lin data size."
    )
    parser.add_argument(
        "--uinsize", default=18403, type=int, help="Uin data size."
    )
    opt = parser.parse_args()

    """============================================================================"""
    opt.source_path += "/" + opt.dataset

    if opt.dataset == "Inaturalist":
        opt.n_epochs = 90
        opt.tau = [40, 70]
        opt.k_vals = [1, 4, 16, 32]

    if opt.dataset == "vehicle_id":
        opt.k_vals = [1, 5]

    if opt.finetune == "true":
        opt.finetune = True
    elif opt.finetune == "false":
        opt.finetune = False

    if opt.get_features == "true":
        opt.get_features = True
    elif opt.get_features == "false":
        opt.get_features = False

    metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
    LOG = aux.LOGGER(opt, metrics_to_log, name="Base", start_new=True)

    """============================================================================"""
    ##################### NETWORK SETUP ##################

    opt.device = torch.device("cuda")
    model = netlib.networkselect(opt)

    # Push to Device
    _ = model.to(opt.device)

    """============================================================================"""
    #################### DATALOADER SETUPS ##################
    # Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
    # The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
    # Is simply using the training set, however running under the same rules as 'testing' dataloader,
    # i.e. no shuffling and no random cropping.
    dataloaders = data.give_dataloaders(
        opt.dataset, opt.trainset, opt.testset, opt
    )
    # Because the number of supervised classes is dataset dependent, we store them after
    # initializing the dataloader
    opt.num_classes = len(dataloaders["training"].dataset.avail_classes)

    if opt.dataset == "Inaturalist":
        eval_params = {
            "dataloader": dataloaders["testing"],
            "model": model,
            "opt": opt,
            "epoch": 0,
        }

    elif opt.dataset == "vehicle_id":
        eval_params = {
            "dataloaders": [
                dataloaders["testing_set1"],
                dataloaders["testing_set2"],
                dataloaders["testing_set3"],
            ],
            "model": model,
            "opt": opt,
            "epoch": 0,
        }

    """============================================================================"""
    ####################evaluation ##################

    results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
