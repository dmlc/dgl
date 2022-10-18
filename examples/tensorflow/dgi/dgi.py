"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import math

import numpy as np
import tensorflow as tf
from gcn import GCN
from tensorflow.keras import layers


class Encoder(layers.Layer):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(
            g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout
        )

    def call(self, features, corrupt=False):
        if corrupt:
            perm = np.random.permutation(self.g.number_of_nodes())
            features = tf.gather(features, perm)
        features = self.conv(features)
        return features


class Discriminator(layers.Layer):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        uinit = tf.keras.initializers.RandomUniform(
            -1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden)
        )
        self.weight = tf.Variable(
            initial_value=uinit(shape=(n_hidden, n_hidden), dtype="float32"),
            trainable=True,
        )

    def call(self, features, summary):
        features = tf.matmul(
            features, tf.matmul(self.weight, tf.expand_dims(summary, -1))
        )
        return features


class DGI(tf.keras.Model):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(
            g, in_feats, n_hidden, n_layers, activation, dropout
        )
        self.discriminator = Discriminator(n_hidden)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

    def call(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        summary = tf.nn.sigmoid(tf.reduce_mean(positive, axis=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(tf.ones(positive.shape), positive)
        l2 = self.loss(tf.zeros(negative.shape), negative)

        return tf.reduce_mean(l1) + tf.reduce_mean(l2)


class Classifier(layers.Layer):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = layers.Dense(n_classes)

    def call(self, features):
        features = self.fc(features)
        return features
