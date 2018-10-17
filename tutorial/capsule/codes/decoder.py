"""Decoder Network

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Decoder(nn.Module):
    """
    Implement Decoder structure in section 4.1, Figure 2 to reconstruct a digit
    from the `DigitCaps` layer representation.

    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.

    This Decoder network is used in training and prediction (testing).
    """

    def __init__(self, num_classes, output_unit_size, input_width,
                 input_height, num_conv_in_channel, cuda_enabled):
        """
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 (or 3072 for CIFAR10) neurons each.
        """
        super(Decoder, self).__init__()

        self.cuda_enabled = cuda_enabled

        fc1_output_size = 512
        fc2_output_size = 1024
        self.fc3_output_size = input_width * input_height * num_conv_in_channel
        self.fc1 = nn.Linear(num_classes * output_unit_size, fc1_output_size) # input dim 10 * 16.
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size, self.fc3_output_size)
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        """
        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the Decoder network, and
        reconstruct a [batch_size, fc3_output_size] size tensor representing the image.

        Args:
            x: [batch_size, 10, 16] The output of the digit capsule.
            target: [batch_size, 10] One-hot MNIST dataset labels.

        Returns:
            reconstruction: [batch_size, fc3_output_size] Tensor of reconstructed images.
        """
        batch_size = target.size(0)

        """
        First, do masking.
        """
        # Method 1: mask with y.
        # Note: we have not implement method 2 which is masking with true label.
        # masked_caps shape: [batch_size, 10, 16, 1]
        masked_caps = utils.mask(x, self.cuda_enabled)

        """
        Second, reconstruct the images with 3 Fully Connected layers.
        """
        # vector_j shape: [batch_size, 160=10*16]
        vector_j = masked_caps.view(x.size(0), -1) # reshape the masked_caps tensor

        # Forward pass of the network
        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out)) # shape: [batch_size, 1024]
        reconstruction = self.sigmoid(self.fc3(fc2_out)) # shape: [batch_size, fc3_output_size]

        assert reconstruction.size() == torch.Size([batch_size, self.fc3_output_size])

        return reconstruction
