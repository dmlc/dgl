import dgl
import numpy as np
import torch
import torch.nn as nn
import tqdm
import os
import gc
import time

from .auto_tuner import get_auto_tuner
from .function_generator import FunctionGenerator
from .data_manager import DataManager
from .custom_dataloader import CustomDataloader
from .utils import get_new_arg_input, update_ret_output
from .ssd import SSDBlockSampler

class InferenceHelperBase():
    def __init__(self, module: nn.Module, device, use_uva = False, debug = False):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._use_uva = use_uva
        self._function_generator = FunctionGenerator(module, debug)
        self._traced = self._function_generator.traced
        self._schema = self._function_generator.get_schema()
        self._funcs = self._function_generator.get_funcs()
        self._data_manager = DataManager(device, use_uva)
        self._debug = debug

    def _trace_output_shape(self, graph, *args):
        if graph.is_homogeneous:
            first_layer_inputs = (dgl.graph(([0], [0]), device=self._device),)
        else:
            data_dict = {}
            for canonical_etype in graph.canonical_etypes:
                data_dict[canonical_etype] = ([0], [0])
            first_layer_inputs = (dgl.heterograph(data_dict, device=self._device),)
        for arg in tuple(args):
            if arg is None:
                raise Exception("arguments could not be None.")
            else:
                first_layer_inputs += (arg[[0]].to(self._device),)
        arg2val_map = {}
        for val, arg_name in zip(first_layer_inputs, self._schema.first_layer_input):
            arg_node = self._schema.name2arg_map[arg_name]
            arg2val_map[arg_node] = val
        ret_shapes = [[] for _ in range(self._schema.layers_count)]
        for layer, func in zip(self._schema.layers, self._funcs):
            new_args = ()
            for arg_node in layer.inputs:
                new_args += (arg2val_map[arg_node],)
            output_vals = func(*new_args)
            if not isinstance(output_vals, tuple):
                output_vals = (output_vals,)
            if len(output_vals) != len(layer.outputs):
                raise Exception("output values not match with layer's output.")
            for val, arg_node in zip(output_vals, layer.outputs):
                if isinstance(val, torch.Tensor):
                    arg2val_map[arg_node] = val
                    ret_shapes[layer.id].append((torch.Tensor, val.size()[1:]))
                else:
                    ret_shapes[layer.id].append((val.__class__, None))
        return ret_shapes

    def compute(self, inference_graph, rets, layer, func):
        raise NotImplementedError()

    def before_inference(self, graph, *args):
        pass

    def after_inference(self):
        pass

    def init_ret(self, arg_node, shape):
        return torch.zeros(shape)

    def clear_ret(self, arg_node):
        del self._data_manager[arg_node]

    def inference(self, inference_graph, *args):
        self.before_inference(inference_graph, *args)
        if self._use_uva:
            for k in list(inference_graph.ndata.keys()):
                inference_graph.ndata.pop(k)
            for k in list(inference_graph.edata.keys()):
                inference_graph.edata.pop(k)

        first_layer_inputs = (inference_graph,) + tuple(args)
        if len(first_layer_inputs) != len(self._schema.first_layer_input):
            raise Exception("layer's input not match with args.")
        for val, arg_name in zip(first_layer_inputs, self._schema.first_layer_input):
            arg_node = self._schema.name2arg_map[arg_name]
            self._data_manager[arg_node] = val
        ret_shapes = self._trace_output_shape(inference_graph, *args)

        for layer, func in zip(self._schema.layers, self._funcs):

            rets = []
            for j, arg_node in enumerate(layer.outputs):
                cls, shape = ret_shapes[layer.id][j]
                if cls == torch.Tensor:
                    ret_shape = (inference_graph.number_of_nodes(),) + tuple(shape)
                    ret = self.init_ret(arg_node, ret_shape)
                else:
                    ret = None
                rets.append(ret)
                self._data_manager[arg_node] = ret

            gc.collect()
            torch.cuda.empty_cache()

            rets = self.compute(inference_graph, rets, layer, func)

            # delete intermediate val
            for arg_node in layer.inputs:
                if arg_node.input_layers[-1] == layer and arg_node.input_layers[0] != self._schema.get_layer(0):
                    self.clear_ret(arg_node)

        outputs = ()
        for name in self._schema.last_layer_output:
            arg_node = self._schema.name2arg_map[name]
            outputs += (self._data_manager[arg_node],)

        self.after_inference()

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class InferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, batch_size, device, num_workers = 4, debug = False):
        super().__init__(module, device, debug=debug)
        self._batch_size = batch_size
        self._num_workers = num_workers

    def compute(self, graph, rets, layer, func):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            torch.arange(graph.number_of_nodes()).to(graph.device),
            sampler,
            batch_size=self._batch_size,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            new_args = get_new_arg_input(layer.inputs, self._data_manager, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
            del output_vals

        return rets


class EdgeControlInferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, max_edge_in_batch, device, num_workers = 4, debug = False):
        super().__init__(module, device, debug=debug)
        self._max_edge_in_batch = max_edge_in_batch
        self._num_workers = num_workers

    def compute(self, graph, rets, layer, func):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = CustomDataloader(
            graph,
            self._max_edge_in_batch,
            sampler,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)

        pbar = tqdm.tqdm(total=graph.number_of_nodes())
        for input_nodes, output_nodes, blocks in dataloader:
            new_args = get_new_arg_input(layer.inputs, self._data_manager, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
            del output_vals
            pbar.update(output_nodes.shape[0])
        pbar.close()

        return rets


class AutoInferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, device, use_uva, free_rate, use_random, debug = False):
        self.free_rate = free_rate
        self.use_random = use_random
        super().__init__(module, device, use_uva, debug)

    def before_inference(self, graph, *args):
        if not self.use_random:
            self.nids = torch.arange(graph.number_of_nodes()).to(graph.device)
        else:
            self.nids = torch.randperm(graph.number_of_nodes()).to(graph.device)
        in_degrees = graph.in_degrees(self.nids).numpy()
        prefix_sum_in_degrees = np.cumsum(in_degrees)
        self.prefix_sum_in_degrees = [0]
        self.prefix_sum_in_degrees.extend(prefix_sum_in_degrees.tolist())
        self.prefix_sum_in_degrees.append(2e18)

    def compute(self, graph, rets, layer, func):

        if self._use_uva:
            self.nids = self.nids.to(self._device)
            self._data_manager.pin_data_inplace(layer)

        auto_tuner = get_auto_tuner(self._device)
        start_max_node = 2000
        start_max_edge = 500000

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = CustomDataloader(
            graph,
            self.nids,
            sampler,
            start_max_node,
            start_max_edge,
            self.prefix_sum_in_degrees,
            device=self._device,
            use_uva=self._use_uva,
            shuffle=False)

        pbar = tqdm.tqdm(total=graph.number_of_nodes())
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            try:
                auto_tuner.reset_state()
                torch.cuda.empty_cache()
                auto_tuner.set_free(self.free_rate)

                new_args = get_new_arg_input(layer.inputs, self._data_manager, input_nodes, 
                    blocks[0], self._device, self._use_uva)

                output_vals = func(*new_args)
                del new_args
                if self._debug:
                    print(blocks[0], "; max memory = ", torch.cuda.max_memory_allocated() // 1024 ** 2, "MB")

                rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
                del output_vals

                auto_tuner.set_max()
                nxt_max_node, nxt_max_edge = auto_tuner.search(blocks[0])
                pbar.update(output_nodes.shape[0])

            except Exception as e:
                print(e)
                nxt_max_node, nxt_max_edge = auto_tuner.break_peak(blocks[0])
                dataloader.reset_batch_node(output_nodes.shape[0])
                gc.collect()

            finally:
                dataloader.modify_max_node(nxt_max_node)
                dataloader.modify_max_edge(nxt_max_edge)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        if self._use_uva:
            self._data_manager.unpin_data_inplace(layer)

        pbar.close()
        return rets


class SSDAutoInferenceHelper(InferenceHelperBase):
    def __init__(self, module: nn.Module, device, use_uva, free_rate, use_random, debug = False):
        self.free_rate = free_rate
        self.use_random = use_random
        super().__init__(module, device, use_uva, debug)

    def before_inference(self, graph, *args):
        if not self.use_random:
            self.nids = torch.arange(graph.number_of_nodes()).to(graph.device)
        else:
            self.nids = torch.randperm(graph.number_of_nodes()).to(graph.device)
        in_degrees = np.load(f"/realssd/{self.dataset_name}/in_degrees.npy")
        prefix_sum_in_degrees = np.cumsum(in_degrees)
        self.prefix_sum_in_degrees = [0]
        self.prefix_sum_in_degrees.extend(prefix_sum_in_degrees.tolist())
        self.prefix_sum_in_degrees.append(2e18)

    def init_ret(self, arg_node, shape):
        return torch.as_tensor(np.memmap(f"/realssd/feat_{arg_node.name}.npy",dtype=np.float32, mode="w+", shape=shape, ))

    def clear_ret(self, arg_node):
        del self._data_manager[arg_node]
        os.remove(f"/realssd/feat_{arg_node.name}.npy")

    def compute(self, graph, rets, layer, func):

        if self._use_uva:
            self.nids = self.nids.to(self._device)
            self._data_manager.pin_data_inplace(layer)

        auto_tuner = get_auto_tuner(self._device)
        start_max_node = 2000
        start_max_edge = 500000

        sampler = SSDBlockSampler()
        dataloader = CustomDataloader(
            graph,
            self.nids,
            sampler,
            start_max_node,
            start_max_edge,
            self.prefix_sum_in_degrees,
            device=self._device,
            use_uva=self._use_uva,
            shuffle=False)

        for input_nodes, output_nodes, blocks in dataloader:
            try:
                auto_tuner.reset_state()
                torch.cuda.empty_cache()
                auto_tuner.set_free(self.free_rate)

                new_args = get_new_arg_input(layer.inputs, self._data_manager, input_nodes, 
                    blocks[0], self._device, self._use_uva)

                output_vals = func(*new_args)
                del new_args
                if self._debug:
                    print(blocks[0], "; max memory = ", torch.cuda.max_memory_allocated() // 1024 ** 2, "MB")

                rets = update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks)
                del output_vals

                auto_tuner.set_max()
                nxt_max_node, nxt_max_edge = auto_tuner.search(blocks[0])

            except Exception as e:
                print(e)
                nxt_max_node, nxt_max_edge = auto_tuner.break_peak(blocks[0])
                dataloader.reset_batch_node(output_nodes.shape[0])
                gc.collect()

            finally:
                dataloader.modify_max_node(nxt_max_node)
                dataloader.modify_max_edge(nxt_max_edge)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        if self._use_uva:
            self._data_manager.unpin_data_inplace(layer)

        return rets
