"""Subgraph samplers"""

from torchdata.datapipes.iter import Mapper
from .link_data_format import LinkDataFormat
import torch


class SubgraphSampler(Mapper):
    """A subgraph sampler.

    It is an iterator equivalent to the following:

    .. code:: python

       for data in datapipe:
           yield _sample(data)

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    fn : callable
        The subgraph sampling function.
    """
    
    def __init__(
        self,
        datapipe,
        fanouts,
        replace,
        prob_name,
    ):
        """
        Initlization for a negative sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        negative_ratio : int
            The proportion of negative samples to positive samples.
        link_data_format : LinkDataFormat
            Determines the format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges,  while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).
        """
        super().__init__(datapipe, self._sample)
        self.fanouts = fanouts
        self.replace = replace
        self.prob_name = prob_name
        
    
    def _sample(self, data):
        adjs = []
        seeds, _ = self._pre_process(data)
        num_layers = len(self.fanouts)
        for hop in range(num_layers):
            seeds, sg = self._generate(seeds, torch.LongTensor(self.fanout[hop]))
            adjs.insert(0, sg)

    def _generate(self, seeds):
        raise NotImplemented


class LinkSubgraphSampler(SubgraphSampler):
    def __init__(
        self,
        datapipe,
        fanouts,
        replace,
        probs_name,
        link_data_format,
    ):
        super.__init__(datapipe, fanouts, replace, probs_name)
        assert link_data_format in [
            LinkDataFormat.CONDITIONED,
            LinkDataFormat.INDEPENDENT,
        ], f"Unsupported data format: {link_data_format}."
        self.link_data_format = link_data_format
        