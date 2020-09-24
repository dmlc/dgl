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

def check_args(args):
    flag = sum([args.only_1st, args.only_2nd])
    assert flag <= 1, "no more than one selection from --only_1st and --only_2nd"
    if flag == 0:
        assert args.dim % 2 == 0, "embedding dimension must be an even number"
    if args.async_update:
        assert args.mix, "please use --async_update with --mix"

def sum_up_params(model):
    """ Count the model parameters """
    n = []
    if model.fst:
        p = model.fst_u_embeddings.weight.cpu().data.numel()
        n.append(p)
        p = model.fst_state_sum_u.cpu().data.numel()
        n.append(p)
    if model.snd:
        p = model.snd_u_embeddings.weight.cpu().data.numel() * 2
        n.append(p)
        p = model.snd_state_sum_u.cpu().data.numel() * 2
        n.append(p)
    n.append(model.lookup_table.cpu().numel())
    try:
        n.append(model.index_emb_negu.cpu().numel() * 2)
    except:
        pass
    print("#params " + str(sum(n)))