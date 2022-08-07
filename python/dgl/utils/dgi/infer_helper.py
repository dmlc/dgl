"""Inference helper."""
# pylint: disable=unnecessary-pass
import dgl
import torch
import torch.nn as nn
import tqdm

from torch.fx import GraphModule
from .utils import get_new_arg_input, update_ret_output
from .splitter import split_module
from .tracer import dgl_symbolic_trace


class InferenceHelperBase():
    """Inference helper base class.

    This class is the base class for inference helper. Users can create an inference
    helper to compute layer-wise inference easily. The inference helper provides a
    simple interence, which users only need to call the ``inference`` function after
    create the InferenceHelper object.

    Parameters
    ----------
    root : torch.nn.Module
        The root model to conduct inference.
    device : torch.device
        The device to conduct inference computation.
    conv_modules : Tuple(class)
        The conv modules' classes.
    use_uva : bool
        Whether store graph and tensors in UVA.
    debug : bool
        Whether display debug messages.
    """
    def __init__(self, root: nn.Module, conv_modules = (), device = "cpu", \
                 use_uva = False, debug = False):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._use_uva = use_uva
        self._debug = debug
        graph_module = dgl_symbolic_trace(root, conv_modules)
        self._tags, self._splitted = split_module(graph_module, debug)
        self._wrap_conv_blocks()

    def _get_mock_graph(self, graph):
        """Get the mock graph."""
        if self._mock_graph is None:
            self._input_graph = graph
            if graph.is_homogeneous:
                self._mock_graph = dgl.graph(([0], [0]), device=self._device)
            else:
                data_dict = {}
                for canonical_etype in graph.canonical_etypes:
                    data_dict[canonical_etype] = ([0], [0])
                self._mock_graph = dgl.heterograph(data_dict, device=self._device)
        return self._mock_graph

    def _trace_output_shape(self, func, *args):
        """Trace the output shape."""
        mock_input = ()
        for arg in args:
            if isinstance(arg, dgl.DGLHeteroGraph):
                mock_input += (self._get_mock_graph(arg),)
            elif isinstance(arg, torch.Tensor):
                mock_input += (arg[[0]].to(self._device),)
            else:
                raise Exception("Input type not supported yet.")

        assert self._input_graph is not None
        mock_rets = func(*mock_input)

        if not isinstance(mock_rets, tuple):
            mock_rets = (mock_rets,)
        ret_shapes = []
        for mock_ret in mock_rets:
            if isinstance(mock_ret, torch.Tensor):
                ret_shapes.append((self._input_graph.number_of_nodes(),) + mock_ret.size()[1:])
            else:
                raise Exception("Output type not supported yet.")
        return ret_shapes

    def _wrap_conv_blocks(self):
        """Wrap Conv blocks to calls."""
        def _warpped_call(self, *args):
            torch.cuda.empty_cache()
            ret_shapes = self.helper._trace_output_shape(self, *args)
            rets = ()
            for ret_shape in ret_shapes:
                rets += (self.helper.init_ret(ret_shape),)

            outputs = self.helper.compute(rets, self, *args)
            if len(outputs) == 1:
                return outputs[0]
            return outputs

        GraphModule.wraped_call = _warpped_call
        for tag in self._tags:
            sub_gm = getattr(self._splitted, tag)
            sub_gm.helper = self
            self._splitted.delete_submodule(tag)
            setattr(self._splitted, tag, sub_gm.wraped_call)

    def compute(self, rets, func, *args):
        """Compute function.

        The abstract function for compute one layer convolution. Inside the inference
        function, the compute function is used for compute the message for next layer
        tensors. Users can override this function for their customize requirements.

        Parameters
        ----------
        rets : Tuple[Tensors]
            The predefined output tensors for this layer.
        func : Callable
            The function for computation.
        *args : Tuple
            The arguments for computing.

        Returns
        ----------
        Tuple[Tensors] or Tensor
            Output tensors.
        """
        raise NotImplementedError()

    def before_inference(self, graph, *args):
        """What users want to do before inference.

        Parameters
        ----------
        graph : DGLHeteroGraph
            The graph object.
        args : Tuple
            The input arguments, same as ``inference`` function.
        """
        pass

    def after_inference(self):
        """What users want to do after inference."""
        pass

    def init_ret(self, shape):
        """The initization for ret.

        Users can override it if customize initization needs. For example use numpy memmap.

        Parameters
        ----------
        shape : Tuple[int]
            The shape of output tensors.

        Returns
        ----------
        Tensor
            Output tensor (empty).
        """
        return torch.zeros(shape)

    def inference(self, inference_graph, *args):
        """The inference function.

        Call the inference function can conduct the layer-wise inference computation.

        inference_graph : DGLHeteroGraph
            The input graph object.
        args : Tuple
            The input arguments, should be the same as module's forward function.
        """
        self.before_inference(inference_graph, *args)
        self._input_graph = None
        self._mock_graph = None
        if self._use_uva:
            for k in list(inference_graph.ndata.keys()):
                inference_graph.ndata.pop(k)
            for k in list(inference_graph.edata.keys()):
                inference_graph.edata.pop(k)

        outputs = self._splitted(inference_graph, *args)

        self.after_inference()

        return outputs


class InferenceHelper(InferenceHelperBase):
    """The InferenceHelper class.

    To construct an inference helper for customized requirements, users can extend the
    InferenceHelperBase class and write their own compute function (which can refer the
    InferenceHelper's implementation).

    Parameters
    ----------
    root : torch.nn.Module
        The model to conduct inference.
    batch_size : int
        The batch size for dataloader.
    device : torch.device
        The device to conduct inference computation.
    conv_modules : Tuple(class)
        The conv modules' classes.
    num_workers : int
        Number of workers for dataloader.
    use_uva : bool
        Whether store graph and tensors in UVA.
    debug : bool
        Whether display debug messages.
    """
    def __init__(self, root: nn.Module, batch_size, device, conv_modules = (), \
                 num_workers = 4, debug = False):
        super().__init__(root, conv_modules, device, debug=debug)
        self._batch_size = batch_size
        self._num_workers = num_workers

    def compute(self, rets, func, *args):
        """Compute function.

        The basic compute function inside the inference helper. Users should not call this
        function on their own.

        Parameters
        ----------
        rets : Tuple[Tensors]
            The predefined output tensors for this layer.
        func : Callable
            The function for computation.
        *args : Tuple
            The arguments for computing.

        Returns
        ----------
        Tuple[Tensors] or Tensor
            Output tensors.
        """
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            self._input_graph,
            torch.arange(self._input_graph.number_of_nodes()).to(self._input_graph.device),
            sampler,
            batch_size=self._batch_size,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            new_args = get_new_arg_input(args, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, output_nodes, blocks)
            del output_vals

        return rets
