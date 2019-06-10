import mxnet.ndarray as F
import numpy as np
import warnings
from mxnet.gluon import nn, HybridBlock, Block
from utils import get_activation
import mxnet as mx

class LayerDictionary(Block):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        input_dims : dict
        output_dims : dict
        """
        super(LayerDictionary, self).__init__(**kwargs)
        self._key2idx = dict()
        with self.name_scope():
            self._layers = nn.Sequential()
        self._nlayers = 0

    def __len__(self):
        return len(self._layers)

    def __setitem__(self, key, layer):
        if key in self._key2idx:
            warnings.warn('Duplicate Key. Need to test the code!')
            self._layers[self._key2idx[key]] = layer
        else:
            self._layers.add(layer)
            self._key2idx[key] = self._nlayers
            self._nlayers += 1

    def __getitem__(self, key):
        return self._layers[self._key2idx[key]]

    def __contains__(self, key):
        return key in self._key2idx


class MultiLinkGCNAggregator(Block):
    def __init__(self, g, src_key, dst_key, units, in_units, num_links,
                 dropout_rate=0.0, accum='stack', act=None, **kwargs):
        super(MultiLinkGCNAggregator, self).__init__(**kwargs)
        self.g = g
        self._src_key = src_key
        self._dst_key = dst_key
        self._accum = accum
        self._num_links = num_links
        if accum == "stack":
            assert units % num_links == 0, 'units should be divisible by the num_links '
            self._units = self._units // num_links
        elif accum == "sum":
            self._units = units
        else:
            raise NotImplementedError

        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            self.act = get_activation(act)
            ### TODO kwargs only be supported in hybridBlock
            # for i in range(num_links):
                # self.__setattr__('weight{}'.format(i),
                #                  self.params.get('weight{}'.format(i),
                #                                  shape=(units, 0),
                #                                  dtype=np.float32,
                #                                  allow_deferred_init=True))
                # self.__setattr__('bias{}'.format(i),
                #                  self.params.get('bias{}'.format(i),
                #                                  shape=(units,),
                #                                  dtype=np.float32,
                #                                  init='zeros',
                #                                  allow_deferred_init=True))
            self.weights = self.params.get('weight',
                                           shape=(num_links, self._units, in_units),
                                           dtype=np.float32,
                                           allow_deferred_init=True)
            # self.biases = self.params.get('bias',
            #                               shape=(num_links, units, ),
            #                               dtype=np.float32,
            #                               init='zeros',
            #                               allow_deferred_init=True)

    def forward(self, src_input, dst_input):
        src_input = self.dropout(src_input)
        dst_input = self.dropout(dst_input)
        #print("self._src_key", self._src_key)
        #print("self._dst_key", self._dst_key)
        self.g[self._src_key].ndata['fea'] = src_input
        self.g[self._dst_key].ndata['fea'] = dst_input
        def message_func(edges):
            #print("\n\n In the message function ...")
            msg_dic = {}
            for i in range(self._num_links):
                # w = kwargs['weight{}'.format(i)]
                w = self.weights.data()[i]
                # print("w", w.shape)
                # print("edges.data['support{}')]".format(i), edges.data['support{}'.format(i)] )
                # print("edges.src['h']", edges.src['h'])
                msg_dic['msg{}'.format(i)] = mx.nd.reshape(edges.data['support{}'.format(i)], shape=(-1, 1))\
                                             * mx.nd.dot(edges.src['fea'], w, transpose_b=True)
            return msg_dic

        def reduce_func(nodes):
            out_l = []
            for i in range(self._num_links):
                # b = kwargs['bias{}'.format(i)]
                # b = self.biases.data()[i]
                out_l.append(mx.nd.sum(nodes.mailbox['msg{}'.format(i)], 1))
            if self._accum == "sum":
                return {'accum': mx.nd.add_n(*out_l)}
            elif self._accum == "stack":
                return {'accum': mx.nd.concat(*out_l, dim=1)}
            else:
                raise NotImplementedError

        def apply_node_func(nodes):
            return {'h': self.act(nodes.data['accum'])}

        self.g.register_message_func(message_func)
        self.g[self._dst_key].register_reduce_func(reduce_func)
        self.g[self._dst_key].register_apply_node_func(apply_node_func)
        self.g.send_and_recv(self.g.edges('uv', 'srcdst'))

        h = self.g[self._dst_key].ndata.pop('h')
        return h

class GCMCLayer(Block):
    def __init__(self, uv_graph, vu_graph, src_key, dst_key, in_units, agg_units, out_units, num_links,
                 dropout_rate=0.0, agg_accum='stack', agg_act=None, out_act=None,
                 agg_ordinal_sharing=False, share_agg_weights=False, share_out_fc_weights=False,
                 **kwargs):
        super(GCMCLayer, self).__init__(**kwargs)
        self._out_act = get_activation(out_act)
        self.uv_graph = uv_graph
        self.vu_graph = vu_graph
        self._src_key = src_key
        self._dst_key = dst_key
        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            self._aggregators = LayerDictionary(prefix='agg_')
            with self._aggregators.name_scope():
                self._aggregators[src_key] = MultiLinkGCNAggregator(g=uv_graph,
                                                                    src_key=src_key,
                                                                    dst_key=dst_key,
                                                                    units = agg_units,
                                                                    in_units=in_units,
                                                                    num_links=num_links,
                                                                    dropout_rate=dropout_rate,
                                                                    accum=agg_accum,
                                                                    act=agg_act,
                                                                    prefix='{}_'.format(src_key))
                self._aggregators[dst_key] = MultiLinkGCNAggregator(g=vu_graph,
                                                                    src_key=dst_key,
                                                                    dst_key=src_key,
                                                                    in_units=in_units,
                                                                    units=agg_units,
                                                                    num_links=num_links,
                                                                    dropout_rate=dropout_rate,
                                                                    accum=agg_accum,
                                                                    act=agg_act,
                                                                    prefix='{}_'.format(dst_key))
            self._out_fcs = LayerDictionary(prefix='out_fc_')
            with self._out_fcs.name_scope():
                self._out_fcs[src_key] = nn.Dense(out_units, flatten=False,
                                                  prefix='{}_'.format(src_key))
                self._out_fcs[dst_key] = nn.Dense(out_units, flatten=False,
                                                  prefix='{}_'.format(dst_key))

            self._out_act = get_activation(out_act)

    def forward(self, user_fea, movie_fea):
        movie_h = self._aggregators[self._src_key](user_fea, movie_fea)
        user_h = self._aggregators[self._dst_key](movie_fea, user_fea)
        out_user = self._out_act(self._out_fcs[self._src_key](user_h))
        out_movie = self._out_act(self._out_fcs[self._dst_key](movie_h))
        return out_user, out_movie


class BiDecoder(HybridBlock):
    def __init__(self, in_units, out_units, num_basis_functions=2, prefix=None, params=None):
        super(BiDecoder, self).__init__(prefix=prefix, params=params)
        self._num_basis_functions = num_basis_functions
        with self.name_scope():
            for i in range(num_basis_functions):
                self.__setattr__('weight{}'.format(i),
                                 self.params.get('weight{}'.format(i), shape=(in_units, in_units),
                                                 init=mx.initializer.Orthogonal(scale=1.1,
                                                                                rand_type='normal'),
                                                 allow_deferred_init=True))
            self.rate_out = nn.Dense(units=out_units, flatten=False, use_bias=False, prefix="rate_")

    def hybrid_forward(self, F, data1, data2, **kwargs):
        basis_outputs_l = []
        for i in range(self._num_basis_functions):
            basis_out = F.sum(F.dot(data1, kwargs["weight{}".format(i)]) * data2,
                              axis=1, keepdims=True)
            basis_outputs_l.append(basis_out)
        basis_outputs = F.concat(*basis_outputs_l, dim=1)
        out = self.rate_out(basis_outputs)
        return out


class InnerProductLayer(HybridBlock):
    def __init__(self, mid_units=None, **kwargs):
        super(InnerProductLayer, self).__init__(**kwargs)
        self._mid_units = mid_units
        if self._mid_units is not None:
            self._mid_map = nn.Dense(mid_units, flatten=False)

    def hybrid_forward(self, F, data1, data2):
        if self._mid_units is not None:
            data1 = self._mid_map(data1)
            data2 = self._mid_map(data2)
        score = F.sum(data1 * data2, axis=1, keepdims=True)
        return score

