import datetime
import os
import random
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init

########################################################################################################################
#                                                    configuration                                                     #
########################################################################################################################


def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
        print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Directory {} already exists.".format(path))
        else:
            raise


def date_filename(base_dir="./"):
    dt = datetime.datetime.now()
    return os.path.join(
        base_dir,
        "{}_{:02d}-{:02d}-{:02d}".format(
            dt.date(), dt.hour, dt.minute, dt.second
        ),
    )


def setup_log_dir(opts):
    log_dir = "{}".format(date_filename(opts["log_dir"]))
    mkdir_p(log_dir)
    return log_dir


def save_arg_dict(opts, filename="settings.txt"):
    def _format_value(v):
        if isinstance(v, float):
            return "{:.4f}".format(v)
        elif isinstance(v, int):
            return "{:d}".format(v)
        else:
            return "{}".format(v)

    save_path = os.path.join(opts["log_dir"], filename)
    with open(save_path, "w") as f:
        for key, value in opts.items():
            f.write("{}\t{}\n".format(key, _format_value(value)))
    print("Saved settings to {}".format(save_path))


def setup(args):
    opts = args.__dict__.copy()

    cudnn.benchmark = False
    cudnn.deterministic = True

    # Seed
    if opts["seed"] is None:
        opts["seed"] = random.randint(1, 10000)
    random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    # Dataset
    from configure import dataset_based_configure

    opts = dataset_based_configure(opts)

    assert (
        opts["path_to_dataset"] is not None
    ), "Expect path to dataset to be set."
    if not os.path.exists(opts["path_to_dataset"]):
        if opts["dataset"] == "cycles":
            from cycles import generate_dataset

            generate_dataset(
                opts["min_size"],
                opts["max_size"],
                opts["ds_size"],
                opts["path_to_dataset"],
            )
        else:
            raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

    # Optimization
    if opts["clip_grad"]:
        assert (
            opts["clip_grad"] is not None
        ), "Expect the gradient norm constraint to be set."

    # Log
    print("Prepare logging directory...")
    log_dir = setup_log_dir(opts)
    opts["log_dir"] = log_dir
    mkdir_p(log_dir + "/samples")

    plt.switch_backend("Agg")

    save_arg_dict(opts)
    pprint(opts)

    return opts


########################################################################################################################
#                                                         model                                                        #
########################################################################################################################


def weights_init(m):
    """
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def dgmg_message_weight_init(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1.0 / 10)
            init.normal_(m.bias.data, std=1.0 / 10)
        else:
            raise ValueError("Expected the input to be of type nn.Linear!")

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    else:
        m.apply(_weight_init)
