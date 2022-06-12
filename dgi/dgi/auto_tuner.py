import pynvml
import torch

def get_auto_tuner(device, cached=0):
    if isinstance(device, torch.device):
        device = device.type
    if 'cuda' in device:
        return GPUAutoTuner(cached)
    elif 'cpu' in device:
        return CPUAutoTuner()
    else:
        raise NotImplementedError("Not implement Auto Tuner for device: {}.".format(device))


class AutoTunerBase:
    def __init__(self):
        self.free_memory = 0
        self.set_free()

    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError

    def search(self, g):
        raise NotImplementedError

    def break_peak(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        return curr_node // 2, curr_edge // 2


class GPUAutoTuner(AutoTunerBase):
    def __init__(self, cached=0):
        self.cached = cached
        self.maxs = []
        super().__init__()

    def reset_state(self):
        self.free_memory = 0
        self.maxs = []

    def set_free(self, rate=0.9):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.free_memory = info.free * rate

    def set_max(self):
        self.maxs.append(torch.cuda.max_memory_allocated() - self.cached)

    def get_max(self):
        return max(self.maxs)

    def search(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        increase_rate = self.free_memory / self.get_max()
        curr_node = int(curr_node * increase_rate)
        curr_edge = int(curr_edge * increase_rate)
        return curr_node, curr_edge

# TODO: Auto inference on CPU.
class CPUAutoTuner(AutoTunerBase):
    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError
