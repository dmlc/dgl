
from torch.utils.data import DataLoader
from functools import wraps

def wrap_output(dataloader, output_wrapper):
    def wrapped_collate_fn(old_collate_fn):
        @wraps(old_collate_fn)
        def new_collate_fn(input_):
            output = old_collate_fn(input_)
            return output_wrapper(*output)
        return new_collate_fn
    dataloader.collate_fn = wrapped_collate_fn(dataloader.collate_fn)
    return dataloader
