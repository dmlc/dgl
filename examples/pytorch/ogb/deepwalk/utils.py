import torch
from functools import wraps
from _thread import start_new_thread
import torch.multiprocessing as mp

def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

def shuffle_walks(walks):
    seeds = torch.randperm(walks.size()[0])
    return walks[seeds]

def sum_up_params(model):
    """ Count the model parameters """
    n = []
    n.append(model.u_embeddings.weight.cpu().data.numel() * 2)
    n.append(model.lookup_table.cpu().numel())
    n.append(model.index_emb_posu.cpu().numel() * 2)
    n.append(model.grad_u.cpu().numel() * 2)

    try:
        n.append(model.index_emb_negu.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.state_sum_u.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.grad_avg.cpu().numel())
    except:
        pass
    try:
        n.append(model.context_weight.cpu().numel())
    except:
        pass

    print("#params " + str(sum(n)))
    exit()