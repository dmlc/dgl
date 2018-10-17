"""Utilities

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Set the logger
writer = SummaryWriter()
step = {'step': 0}


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def checkpoint(state, epoch):
    """Save checkpoint"""
    model_out_path = 'results/trained_model/model_epoch_{}.pth'.format(epoch)
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))


def load_mnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    # Normalize MNIST dataset.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading MNIST training datasets')
    # MNIST dataset
    training_set = datasets.MNIST(
        './data', train=True, download=True, transform=data_transform)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading MNIST testing datasets')
    testing_set = datasets.MNIST(
        './data', train=False, download=True, transform=data_transform)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


def load_cifar10(args):
    """Load CIFAR10 dataset.
    The data is split and normalized between train and test sets.
    """
    # Normalize CIFAR10 dataset.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading CIFAR10 training datasets')
    # CIFAR10 dataset
    training_set = datasets.CIFAR10(
        './data', train=True, download=True, transform=data_transform)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading CIFAR10 testing datasets')
    testing_set = datasets.CIFAR10(
        './data', train=False, download=True, transform=data_transform)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


def load_data(args):
    """
    Load dataset.
    """
    dst = args.dataset

    if dst == 'mnist':
        return load_mnist(args)
    elif dst == 'cifar10':
        return load_cifar10(args)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dst)


def squash(sj, dim=2):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0

    This implement equation 1 from the paper.
    """
    sj_mag_sq = torch.sum(sj ** 2, dim, keepdim=True)
    # ||sj||
    sj_mag = torch.sqrt(sj_mag_sq)
    v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
    return v_j


def mask(out_digit_caps, cuda_enabled=True):
    """
    In the paper, they mask out all but the activity vector of the correct digit capsule.

    This means:
    a) during training, mask all but the capsule (1x16 vector) which match the ground-truth.
    b) during testing, mask all but the longest capsule (1x16 vector).

    Args:
        out_digit_caps: [batch_size, 10, 16] Tensor output of `DigitCaps` layer.

    Returns:
        masked: [batch_size, 10, 16, 1] The masked capsules tensors.
    """
    # a) Get capsule outputs lengths, ||v_c||
    v_length = torch.sqrt((out_digit_caps ** 2).sum(dim=2))

    # b) Pick out the index of longest capsule output, v_length by
    # masking the tensor by the max value in dim=1.
    _, max_index = v_length.max(dim=1)
    max_index = max_index.data

    # Method 1: masking with y.
    # c) In all batches, get the most active capsule
    # It's not easy to understand the indexing process with max_index
    # as we are 3D animal.
    batch_size = out_digit_caps.size(0)
    masked_v = [None] * batch_size  # Python list
    for batch_ix in range(batch_size):
        # Batch sample
        sample = out_digit_caps[batch_ix]

        # Masks out the other capsules in this sample.
        v = Variable(torch.zeros(sample.size()))
        if cuda_enabled:
            v = v.cuda()

        # Get the maximum capsule index from this batch sample.
        max_caps_index = max_index[batch_ix]
        v[max_caps_index] = sample[max_caps_index]
        masked_v[batch_ix] = v  # append v to masked_v

    # Concatenates sequence of masked capsules tensors along the batch dimension.
    masked = torch.stack(masked_v, dim=0)

    return masked


def save_image(image, file_name):
    """
    Save a given image into an image file
    """
    # Check number of channels in an image.
    if image.size(1) == 2:
        # 2-channel image
        zeros = torch.zeros(image.size(0), 1, image.size(2), image.size(3))
        image_tensor = torch.cat([zeros, image.data.cpu()], dim=1)
    else:
        # Grayscale or RGB image
        image_tensor = image.data.cpu()  # get Tensor from Variable

    vutils.save_image(image_tensor, file_name)


def accuracy(output, target, cuda_enabled=True):
    """
    Compute accuracy.

    Args:
        output: [batch_size, 10, 16, 1] The output from DigitCaps layer.
        target: [batch_size] Labels for dataset.

    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)

    v_length = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
    softmax_v = F.softmax(v_length, dim=1)
    assert softmax_v.size() == torch.Size([batch_size, 10, 1, 1])

    _, max_index = softmax_v.max(dim=1)
    assert max_index.size() == torch.Size([batch_size, 1, 1])

    pred = max_index.squeeze()  # max_index.view(batch_size)
    assert pred.size() == torch.Size([batch_size])

    if cuda_enabled:
        target = target.cuda()
        pred = pred.cuda()

    correct_pred = torch.eq(target, pred.data)  # tensor
    # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
    acc = correct_pred.float().mean()  # e.g: 6 / 128 = 0.046875

    return acc


def to_np(param):
    """
    Convert values of the model parameters to numpy.array.
    """
    return param.clone().cpu().data.numpy()


def str2bool(v):
    """
    Parsing boolean values with argparse.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
