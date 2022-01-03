
class TensorStorage(object):
    def __init__(self, tensor):
        self.tensor = tensor
        self.feature_shape = tensor.shape[1:]
        self.is_cuda = (tensor.device.type == 'cuda')

    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        event = None
        if not self.is_cuda:
            # CPU to CPU or CUDA - use pin_memory and async transfer if possible
            result = torch.empty(
                indices.shape[0], *self.feature_shape, pin_memory=pin_memory)
            torch.index_select(self.tensor, 0, indices, out=result)
            if device.type == 'cuda':
                result = result.to(device, non_blocking=True)
            else:
                result = result.to(device)
        else:
            # CUDA to CUDA or CPU
            result = torch.index_select(self.tensor, 0, indices).to(device)
        return result
