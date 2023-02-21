# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

"""to do:

clean all of the files - particularly the main.py and also the losses and dataset files and the file for doing the dataloading

-- fast loading etc

need to change all of the copyrights at the top of all of the files

"""

#################### LIBRARIES ########################
import warnings

warnings.filterwarnings("ignore")

import argparse
import datetime
import os
import random

import matplotlib
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))
matplotlib.use("agg")

import auxiliaries as aux
import datasets as data
import evaluate as eval
import losses as losses
import netlib as netlib
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()
####### Main Parameter: Dataset to use for Training
parser.add_argument(
    "--dataset",
    default="Inaturalist",
    type=str,
    help="Dataset to use.",
    choices=["Inaturalist", "semi_fungi"],
)
### General Training Parameters
parser.add_argument(
    "--lr",
    default=0.00001,
    type=float,
    help="Learning Rate for network parameters.",
)
parser.add_argument(
    "--fc_lr_mul",
    default=5,
    type=float,
    help="OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.",
)
parser.add_argument(
    "--n_epochs", default=400, type=int, help="Number of training epochs."
)
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
    help="Number of samples in one class drawn before choosing the next class",
)
parser.add_argument(
    "--seed", default=1, type=int, help="Random seed for reproducibility."
)
parser.add_argument(
    "--scheduler",
    default="step",
    type=str,
    help="Type of learning rate scheduling. Currently: step & exp.",
)
parser.add_argument(
    "--gamma",
    default=0.3,
    type=float,
    help="Learning rate reduction after tau epochs.",
)
parser.add_argument(
    "--decay", default=0.0004, type=float, help="Weight decay for optimizer."
)
parser.add_argument(
    "--tau",
    default=[200, 300],
    nargs="+",
    type=int,
    help="Stepsize(s) before reducing learning rate.",
)
parser.add_argument(
    "--infrequent_eval",
    default=0,
    type=int,
    help="only compute evaluation metrics every 10 epochs",
)
parser.add_argument("--opt", default="adam", help="adam or sgd")
##### Loss-specific Settings
parser.add_argument("--loss", default="smoothap", type=str)
parser.add_argument(
    "--sigmoid_temperature",
    default=0.01,
    type=float,
    help="SmoothAP: the temperature of the sigmoid used in SmoothAP loss",
)
##### Evaluation Settings
parser.add_argument(
    "--k_vals",
    nargs="+",
    default=[1, 2, 4, 8],
    type=int,
    help="Recall @ Values.",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    help="path to checkpoint to load weights from (if empty then ImageNet pre-trained weights are loaded",
)
##### Network parameters
parser.add_argument(
    "--embed_dim",
    default=512,
    type=int,
    help="Embedding dimensionality of the network",
)
parser.add_argument(
    "--arch",
    default="resnet50",
    type=str,
    help="Network backend choice: resnet50, googlenet, BNinception",
)
parser.add_argument(
    "--grad_measure",
    action="store_true",
    help="If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.",
)
parser.add_argument(
    "--dist_measure",
    action="store_true",
    help="If added, the ratio between intra- and interclass distances is stored after each epoch.",
)
parser.add_argument(
    "--not_pretrained",
    action="store_true",
    help="If added, the network will be trained WITHOUT ImageNet-pretrained weights.",
)
##### Setup Parameters
parser.add_argument("--gpu", default=0, type=int, help="GPU-id for GPU to use.")
parser.add_argument(
    "--savename",
    default="",
    type=str,
    help="Save folder name if any special information is to be included.",
)
### Paths to datasets and storage folder
parser.add_argument(
    "--source_path",
    default="/scratch/shared/beegfs/abrown/datasets",
    type=str,
    help="Path to data",
)
parser.add_argument(
    "--save_path",
    default=os.getcwd() + "/Training_Results",
    type=str,
    help="Where to save the checkpoints",
)
### adational
parser.add_argument("--trainset", default="lin_train_set1.txt", type=str)
parser.add_argument("--all_trainset", default="train_set1.txt", type=str)
parser.add_argument("--testset", default="test_set1.txt", type=str)
parser.add_argument("--finetune", default="true", type=str)
parser.add_argument("--cluster_path", default="", type=str)
parser.add_argument("--get_features", default="false", type=str)
parser.add_argument("--class_num", default=948, type=int)
parser.add_argument("--iter", default=0, type=int)
parser.add_argument(
    "--pretrained_weights", default="", type=str, help="pretrained weight path"
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
    "--drop_path_rate", default=0.1, type=float, help="stochastic depth rate"
)
parser.add_argument("--linsize", default=29011, type=int, help="Lin data size.")
parser.add_argument("--uinsize", default=18403, type=int, help="Uin data size.")
opt = parser.parse_args()
"""============================================================================"""
opt.source_path += "/" + opt.dataset
opt.save_path += "/" + opt.dataset + "_" + str(opt.embed_dim)

if opt.dataset == "Inaturalist":
    opt.n_epochs = 90
    opt.tau = [40, 70]
    opt.k_vals = [1, 4, 16, 32]

if opt.dataset == "semi_fungi":
    opt.tau = [40, 70]
    opt.k_vals = [1, 4, 16, 32]

if opt.get_features == "true":
    opt.get_features = True
if opt.get_features == "false":
    opt.get_features = False

if opt.finetune == "true":
    opt.finetune = True
elif opt.finetune == "false":
    opt.finetune = False

"""==========================================================================="""
################### TensorBoard Settings ##################
timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
exp_name = aux.args2exp_name(opt)
opt.save_name = f"weights_{exp_name}" + "/" + timestamp
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

"""============================================================================"""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)
print("using #GPUs:", torch.cuda.device_count())

"""============================================================================"""
##################### NETWORK SETUP ##################

opt.device = torch.device("cuda")
model = netlib.networkselect(opt)

# Push to Device
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
_ = model.to(opt.device)

# Place trainable parameter in list of parameters to train:

if "fc_lr_mul" in vars(opt).keys() and opt.fc_lr_mul != 0:
    all_but_fc_params = list(
        filter(lambda x: "last_linear" not in x[0], model.named_parameters())
    )

    for ind, param in enumerate(all_but_fc_params):
        all_but_fc_params[ind] = param[1]

    if torch.cuda.device_count() > 1:
        fc_params = model.module.model.last_linear.parameters()
    else:
        fc_params = model.model.last_linear.parameters()

    to_optim = [
        {"params": all_but_fc_params, "lr": opt.lr, "weight_decay": opt.decay},
        {
            "params": fc_params,
            "lr": opt.lr * opt.fc_lr_mul,
            "weight_decay": opt.decay,
        },
    ]
else:
    to_optim = [
        {"params": model.parameters(), "lr": opt.lr, "weight_decay": opt.decay}
    ]
"""============================================================================"""
#################### DATALOADER SETUPS ##################
# Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
# The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
# Is simply using the training set, however running under the same rules as 'testing' dataloader,
# i.e. no shuffling and no random cropping.
dataloaders = data.give_dataloaders(opt.dataset, opt.trainset, opt.testset, opt)
# Because the number of supervised classes is dataset dependent, we store them after
# initializing the dataloader
opt.num_classes = len(dataloaders["training"].dataset.avail_classes)

"""============================================================================"""
#################### CREATE LOGGING FILES ###############
# Each dataset usually has a set of standard metrics to log. aux.metrics_to_examine()
# returns a dict which lists metrics to log for training ('train') and validation/testing ('val')

metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
# example output: {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
#                  'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}

# Using the provided metrics of interest, we generate a LOGGER instance.
# Note that 'start_new' denotes that a new folder should be made in which everything will be stored.
# This includes network weights as well.
# If graphviz is installed on the system, a computational graph of the underlying
# network will be made as well.

"""============================================================================"""
#################### LOSS SETUP ####################
# Depending on opt.loss and opt.sampling, the respective criterion is returned,
# and if the loss has trainable parameters, to_optim is appended.
LOG = aux.LOGGER(opt, metrics_to_log, name="Base", start_new=True)
criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
_ = criterion.to(opt.device)

"""============================================================================"""
##################### OPTIONAL EVALUATIONS #####################
# Store the averaged gradients returned from the embedding to the last conv. layer.
if opt.grad_measure:
    grad_measure = eval.GradientMeasure(opt, name="baseline")
# Store the relative distances between average intra- and inter-class distance.
if opt.dist_measure:
    # Add a distance measure for training distance ratios
    distance_measure = eval.DistanceMeasure(
        dataloaders["evaluation"], opt, name="Train", update_epochs=1
    )
    # #If uncommented: Do the same for the test set
    # distance_measure_test = eval.DistanceMeasure(dataloaders['testing'], opt, name='Train', update_epochs=1)

"""============================================================================"""
#################### OPTIM SETUP ####################
# As optimizer, Adam with standard parameters is used.
if opt.opt == "adam":
    optimizer = torch.optim.Adam(to_optim)
elif opt.opt == "sgd":
    optimizer = torch.optim.SGD(to_optim)
else:
    raise Exception("unknown optimiser")
# for the SOA measures in the paper - need to use SGD and 0.05 learning rate
# optimizer    = torch.optim.Adam(to_optim)
# optimizer    = torch.optim.SGD(to_optim)
if opt.scheduler == "exp":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=opt.gamma
    )
elif opt.scheduler == "step":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.tau, gamma=opt.gamma
    )
elif opt.scheduler == "none":
    print("No scheduling used!")
else:
    raise Exception("No scheduling option for input: {}".format(opt.scheduler))


def same_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


"""============================================================================"""
"""================================ TESTING ==================================="""
"""============================================================================"""
################### SCRIPT MAIN ##########################
print("\n-----\n")
# Compute Evaluation metrics, print them and store in LOG.

_ = model.eval()
aux.vis(
    model,
    dataloaders["training"],
    opt.device,
    split="T_train_iter" + str(opt.iter) + "_" + str(opt.loss),
    opt=opt,
)
aux.vis(
    model,
    dataloaders["testing"],
    opt.device,
    split="all_train_iter" + str(opt.iter) + "_" + str(opt.loss),
    opt=opt,
)
aux.vis(
    model,
    dataloaders["eval"],
    opt.device,
    split="test_iter" + str(opt.iter) + "_" + str(opt.loss),
    opt=opt,
)
# Update the Metric Plot and save it.
print("\n-----\n")
