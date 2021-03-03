""" featgen.py

Node feature generators.

"""
import networkx as nx
import numpy as np
import random

import abc


class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass


class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        print ('feat_dict[0]["feat"]:', feat_dict[0]['feat'].dtype)
        nx.set_node_attributes(G, feat_dict)
        print ('G.nodes[0]["feat"]:', G.nodes[0]['feat'].dtype)


class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {
                i: {"feat": feat[i]} for i in range(feat.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)


class GridFeatureGen(FeatureGen):
    """Grid Feature class."""
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu                    # Mean
        self.sigma = sigma              # Variance
        self.com_choices = com_choices  # List of possible community labels

    def gen_node_features(self, G):
        # Generate community assignment
        community_dict = {
            n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1]
            for n in G.nodes()
        }

        # Generate random variable
        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        # Generate features
        feat_dict = {
            n: {"feat": np.asarray([community_dict[n], s[i]])}
            for i, n in enumerate(G.nodes())
        }

        nx.set_node_attributes(G, feat_dict)
        return community_dict

