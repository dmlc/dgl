# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

############################ LIBRARIES ######################################
from collections import OrderedDict
import os
import torch
import torch.nn as nn
import pretrainedmodels as ptm
import vision_transformer as vits
from vision_transformer import DINOHead
from torchvision import models as torchvision_models
import auxiliaries as aux

import pdb

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
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
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
    if opt.arch == 'resnet50':
        network = ResNet50(opt)
    elif opt.arch == 'vit_base':
        print("use vit_base arch.")
        network = ViT(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))

    if opt.resume:
        if opt.resume != "MoCoPretrained/checkpoint.pth" and opt.resume != "MoCoPretrained/semi_fungi_imagenet_inout.pth.tar":
            opt.save_path = "/home/ubuntu/code/Smooth_AP/src/Training_Results/Inaturalist_{}".format(opt.embed_dim)
            weights = torch.load(os.path.join(opt.save_path, opt.resume))
            weights_state_dict = weights['state_dict']

            if torch.cuda.device_count() > 1:
                encoder_state_dict = OrderedDict()
                for k, v in weights_state_dict.items():
                    k = k.replace('module.', '')
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
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)
            if opt.resume == "MoCoPretrained/checkpoint.pth":
                content = torch.load(os.path.join(opt.save_path, opt.resume))
                weights_state_dict = content['model']

                encoder_state_dict = OrderedDict()
                for k, v in weights_state_dict.items():
                    k = k.replace('module.', '')
                    k = k.replace('encoder.', '')
                    encoder_state_dict[k] = v
                self.model.load_state_dict(encoder_state_dict, strict=False)

            if opt.resume == "MoCoPretrained/semi_fungi_imagenet_inout.pth.tar":
                weights = torch.load(os.path.join(opt.save_path, opt.resume))
                weights_state_dict = weights['model_state_dict']

                encoder_state_dict = OrderedDict()
                for k, v in weights_state_dict.items():
                    k = k.replace('module.', '')
                    k = k.replace('encoder.', '')
                    encoder_state_dict[k] = v
                self.model.load_state_dict(encoder_state_dict, strict=False)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        if opt.embed_dim != 2048:
            self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.loss = opt.loss

        if opt.loss == 'ce' or opt.loss == 'ams' or opt.loss == 'softams':
            self.linear = nn.Linear(opt.embed_dim, opt.class_num)
            self.feature = False
        else:
            self.feature = True

    def forward(self, x, feature=False, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.pars.embed_dim != 2048:
            mod_x = self.model.last_linear(x)
        else:
            mod_x = x

        feat = torch.nn.functional.normalize(mod_x, dim=-1)

        if feature or self.loss == 'smoothap' or self.loss == 'softsmoothap':
            return feat
        else:
            pred = self.linear(feat)
            return pred


"""============================================================="""


def ViT(args):
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        finetune_network = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = finetune_network.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        finetune_network = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                          pretrained=False, drop_path_rate=args.drop_path_rate)
        embed_dim = finetune_network.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        finetune_network = torchvision_models.__dict__[args.arch]()
        embed_dim = finetune_network.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    print("os.path.isfile(args.pretrained_weights)", os.path.isfile(args.pretrained_weights))
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if (
                args.checkpoint_key is not None
                and args.checkpoint_key in state_dict
        ):
            print(
                f"Take key {args.checkpoint_key} in provided checkpoint dict"
            )
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = finetune_network.load_state_dict(state_dict, strict=False)
        if args.arch == "vit_base" and args.patch_size == 16:
            ct = 0
            for child in finetune_network.children():
                ct += 1
                if ct < 3:
                    for param in child.parameters():
                        param.requires_grad = False
                elif ct == 3:
                    subct = 0
                    for child_child in child.children():
                        subct += 1
                        if subct < 12:
                            for param in child_child.parameters():
                                param.requires_grad = False

        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )

    # multi-crop wrapper handles forward with inputs of different resolutions
    # finetune_network = aux.MultiCropWrapper(finetune_network, DINOHead(
    #     embed_dim,
    #     args.embed_dim,
    #     use_bn=args.use_bn_in_head,
    #     norm_last_layer=args.norm_last_layer,
    # ))
    finetune_network = aux.MultiCropWrapper(finetune_network, None)

    return finetune_network
