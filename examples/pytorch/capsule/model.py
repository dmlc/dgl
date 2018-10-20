"""CapsNet Architecture

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from capsule_layer import CapsuleLayer
from conv_layer import ConvLayer
from decoder import Decoder
from dgl_capsule_batch import DGLBatchCapsuleLayer


class Net(nn.Module):
    """
    A simple CapsNet with 3 layers
    """

    def __init__(self, num_conv_in_channel, num_conv_out_channel, num_primary_unit,
                 primary_unit_size, num_classes, output_unit_size, num_routing,
                 use_reconstruction_loss, regularization_scale, input_width, input_height,
                 cuda_enabled):
        """
        In the constructor we instantiate one ConvLayer module and two CapsuleLayer modules
        and assign them as member variables.
        """
        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled

        # Configurations used for image reconstruction.
        self.use_reconstruction_loss = use_reconstruction_loss
        # Input image size and number of channel.
        # By default, for MNIST, the image width and height is 28x28
        # and 1 channel for black/white.
        self.image_width = input_width
        self.image_height = input_height
        self.image_channel = num_conv_in_channel

        # Also known as lambda reconstruction. Default value is 0.0005.
        # We use sum of squared errors (SSE) similar to paper.
        self.regularization_scale = regularization_scale

        # Layer 1: Conventional Conv2d layer.
        self.conv1 = ConvLayer(in_channel=num_conv_in_channel,
                               out_channel=num_conv_out_channel,
                               kernel_size=9)

        # PrimaryCaps
        # Layer 2: Conv2D layer with `squash` activation.
        self.primary = CapsuleLayer(in_unit=0,
                                    in_channel=num_conv_out_channel,
                                    num_unit=num_primary_unit,
                                    unit_size=primary_unit_size,  # capsule outputs
                                    use_routing=False,
                                    num_routing=num_routing,
                                    cuda_enabled=cuda_enabled)

        # DigitCaps
        # Final layer: Capsule layer where the routing algorithm is.
        self.digits = DGLBatchCapsuleLayer(in_unit=num_primary_unit,
                                           in_channel=primary_unit_size,
                                           num_unit=num_classes,
                                           unit_size=output_unit_size,  # 16D capsule per digit class
                                           num_routing=num_routing,
                                           cuda_enabled=cuda_enabled)

        # Reconstruction network
        if use_reconstruction_loss:
            self.decoder = Decoder(num_classes, output_unit_size, input_width,
                                   input_height, num_conv_in_channel, cuda_enabled)

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        """
        # x shape: [128, 1, 28, 28]. 128 is for the batch size.
        # out_conv1 shape: [128, 256, 20, 20]
        out_conv1 = self.conv1(x)
        # out_primary_caps shape: [128, 8, 1152].
        # Total PrimaryCapsules has [32 × 6 × 6 = 1152] capsule outputs.
        out_primary_caps = self.primary(out_conv1)
        # out_digit_caps shape: [128, 10, 16, 1]
        # batch size: 128, 10 digit class, 16D capsule per digit class.
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, image, out_digit_caps, target, size_average=True):
        """Custom loss function

        Args:
            image: [batch_size, 1, 28, 28] MNIST samples.
            out_digit_caps: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
            target: [batch_size, 10] One-hot MNIST dataset labels.
            size_average: A boolean to enable mean loss (average loss over batch size).

        Returns:
            total_loss: A scalar Variable of total loss.
            m_loss: A scalar of margin loss.
            recon_loss: A scalar of reconstruction loss.
        """
        recon_loss = 0
        m_loss = self.margin_loss(out_digit_caps, target)
        if size_average:
            m_loss = m_loss.mean()

        total_loss = m_loss

        if self.use_reconstruction_loss:
            # Reconstruct the image from the Decoder network
            reconstruction = self.decoder(out_digit_caps, target)
            recon_loss = self.reconstruction_loss(reconstruction, image)

            # Mean squared error
            if size_average:
                recon_loss = recon_loss.mean()

            # In order to keep in line with the paper,
            # they scale down the reconstruction loss by 0.0005
            # so that it does not dominate the margin loss.
            total_loss = m_loss + recon_loss * self.regularization_scale

        return total_loss, m_loss, (recon_loss * self.regularization_scale)

    def margin_loss(self, input, target):
        """
        Class loss

        Implement equation 4 in section 3 'Margin loss for digit existence' in the paper.

        Args:
            input: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
            target: target: [batch_size, 10] One-hot MNIST labels.

        Returns:
            l_c: A scalar of class loss or also know as margin loss.
        """
        batch_size = input.size(0)

        # ||vc|| also known as norm.
        v_c = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1) ** 2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1) ** 2
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        return l_c

    def reconstruction_loss(self, reconstruction, image):
        """
        The reconstruction loss is the sum of squared differences between
        the reconstructed image (outputs of the logistic units) and
        the original image (input image).

        Implement section 4.1 'Reconstruction as a regularization method' in the paper.

        Based on naturomics's implementation.

        Args:
            reconstruction: [batch_size, 784] Decoder outputs of reconstructed image tensor.
            image: [batch_size, 1, 28, 28] MNIST samples.

        Returns:
            recon_error: A scalar Variable of reconstruction loss.
        """

        # Calculate reconstruction loss.
        batch_size = image.size(0)  # or another way recon_img.size(0)
        # error = (recon_img - image).view(batch_size, -1)
        image = image.view(batch_size, -1)  # flatten 28x28 by reshaping to [batch_size, 784]
        error = reconstruction - image
        squared_error = error ** 2

        # Scalar Variable
        recon_error = torch.sum(squared_error, dim=1)

        return recon_error
