#### Miscellaneous functions

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
#
# TODO: confirm if this is necessary for MXNet and Tensorflow.  If so, we need
# to standardize worker process creation since our operators are implemented with
# OpenMP.

import torch.multiprocessing as mp
from _thread import start_new_thread
from functools import wraps
import subprocess
import traceback

def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
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

# Get user id
def mps_get_user_id():
    result = subprocess.run(['id', '-u'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').rstrip()

# Start MPS daemon
def mps_daemon_start():
    result = subprocess.run(['nvidia-cuda-mps-control', '-d'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8').rstrip())

# Start MPS server with user id
def mps_server_start(user_id):
    ps = subprocess.Popen(('echo', 'start_server -uid ' + user_id), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()

# Get created server pid
def mps_get_server_pid():
    ps = subprocess.Popen(('echo', 'get_server_list'), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()
    return output.decode('utf-8').rstrip()

# Set active thread percentage with the pid for producer
def mps_set_active_thread_percentage(server_pid, percentage):
    ps = subprocess.Popen(('echo', 'set_active_thread_percentage ' + server_pid + ' ' + str(percentage)), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()
    print('Setting set_active_thread_percentage to', output.decode('utf-8').rstrip())

# Quit MPS
def mps_quit():
    ps = subprocess.Popen(('echo', 'quit'), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()