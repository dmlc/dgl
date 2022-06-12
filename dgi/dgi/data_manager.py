import pynvml
import torch
from dgl.utils import pin_memory_inplace, unpin_memory_inplace


class DataManager:
    def __init__(self, device, use_uva):
        self.arg2val_map = {}
        self.device = device
        self.use_uva = use_uva

    def __getitem__(self, arg_node):
        if arg_node not in self.arg2val_map:
            raise RuntimeError("schema not match with output.")
        return self.arg2val_map[arg_node]

    def __setitem__(self, arg_node, val):
        self.arg2val_map[arg_node] = val

    def __delitem__(self, arg_node):
        del self.arg2val_map[arg_node]

    def pin_data_inplace(self, layer):
        for arg_node in layer.inputs:
            if isinstance(self[arg_node], torch.Tensor) and self[arg_node].device.type == 'cpu':
                pin_memory_inplace(self[arg_node])

    def unpin_data_inplace(self, layer):
        for arg_node in layer.inputs:
            if isinstance(self[arg_node], torch.Tensor) and self[arg_node].device.type == 'cpu':
                unpin_memory_inplace(self[arg_node])


# TODO: we still not use GPU memory for storage yet.
class AutoDataManager(DataManager):
    def __init__(self, device, use_uva):
        super().__init__(device, use_uva)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.tot_free = info.free // 10 # use 10% as storage in GPU
        self.curr = 0
        self.arg_in_gpu = {}

    def __delitem__(self, arg_node):
        if arg_node not in self.arg2val_map:
            return
        self.remove_from_gpu(arg_node)
        super().__delitem__(arg_node)

    def layer_feat_to_gpu(self, layer):
        for arg_node in layer.inputs:
            self.feat_to_gpu(arg_node)
        for arg_node in layer.outputs:
            self.feat_to_gpu(arg_node)

    def layer_remove_from_gpu(self, layer):
        next_layer_input = [] if layer.next() is None else layer.next().inputs
        for arg_node in layer.inputs:
            if arg_node.input_layers[-1] == layer and arg_node.input_layers[0].id != 0:
                del self[arg_node]
        for arg_node in list(self.arg_in_gpu.keys()):
            if arg_node not in next_layer_input:
                self.remove_from_gpu(arg_node)
                self[arg_node] = self[arg_node].cpu()

    def feat_to_gpu(self, arg_node):
        if not isinstance(self[arg_node], torch.Tensor):
            return
        if arg_node not in self.arg_in_gpu:
            memory_comsuption = 4 # float, TODO
            for dim in self[arg_node].shape:
                memory_comsuption *= dim
            if self.curr + memory_comsuption < self.tot_free:
                self[arg_node] = self[arg_node].to(self.device)
                self.arg_in_gpu[arg_node] = memory_comsuption
                self.curr += memory_comsuption
                print("move {} to gpu, comsuption={}, curr={}".format(arg_node.name, memory_comsuption, self.curr) )

    def remove_from_gpu(self, arg_node):
        if not isinstance(self[arg_node], torch.Tensor) or self[arg_node].device.type == 'cpu':
            return
        self.curr -= self.arg_in_gpu[arg_node]
        print("remove {} from gpu, comsuption={}, curr={}".format(arg_node.name, self.arg_in_gpu[arg_node], self.curr) )
        del self.arg_in_gpu[arg_node]
