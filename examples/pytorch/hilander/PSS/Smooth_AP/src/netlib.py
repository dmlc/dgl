# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

############################ LIBRARIES ######################################
import os
from collections import OrderedDict

import auxiliaries as aux
import pretrainedmodels as ptm
import torch
import torch.nn as nn

"""============================================================="""


def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.

    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            module.bias.data.zero_()


"""=================================================================================================================================="""


### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""


### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == "resnet50":
        network = ResNet50(opt)
    else:
        raise Exception("Network {} not available!".format(opt.arch))

    if opt.resume:
        weights = torch.load(os.path.join(opt.save_path, opt.resume))
        weights_state_dict = weights["state_dict"]

        if torch.cuda.device_count() > 1:
            encoder_state_dict = OrderedDict()
            for k, v in weights_state_dict.items():
                k = k.replace("module.", "")
                encoder_state_dict[k] = v

            network.load_state_dict(encoder_state_dict)
        else:
            network.load_state_dict(weights_state_dict)

    # print("=================== network =======================")
    # for parameter in network.parameters():
    #     parameter.requires_grad = False
    # for parameter in network.layer_blocks[-1].parameters():
    #     parameter.requires_grad = True

    return network


"""============================================================="""


class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print("Getting pretrained weights...")
            self.model = ptm.__dict__["resnet50"](
                num_classes=1000, pretrained="imagenet"
            )
            print("Done.")
        else:
            print("Not utilizing pretrained weights!")
            self.model = ptm.__dict__["resnet50"](
                num_classes=1000, pretrained=None
            )
        for module in filter(
            lambda m: type(m) == nn.BatchNorm2d, self.model.modules()
        ):
            module.eval()
            module.train = lambda _: None

        if opt.embed_dim != 2048:
            self.model.last_linear = torch.nn.Linear(
                self.model.last_linear.in_features, opt.embed_dim
            )

        self.layer_blocks = nn.ModuleList(
            [
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ]
        )
        self.loss = opt.loss
        self.feature = True

    def forward(self, x, feature=False, is_init_cluster_generation=False):
        x = self.model.maxpool(
            self.model.relu(self.model.bn1(self.model.conv1(x)))
        )

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.pars.embed_dim != 2048:
            mod_x = self.model.last_linear(x)
        else:
            mod_x = x

        feat = torch.nn.functional.normalize(mod_x, dim=-1)

        if feature or self.loss == "smoothap":
            return feat
        else:
            pred = self.linear(feat)
            return pred
