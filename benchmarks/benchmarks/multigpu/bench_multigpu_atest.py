import time
import numpy as np
import torch as th
import torch.multiprocessing as mp


def run(proc_id, n_gpus, n_cpus, devices, queue=None):
    dev_id = devices[proc_id]
    world_size = n_gpus
    data = th.rand(1000, 1000).cuda(dev_id)
    print(data.shape)

    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    backend = "nccl"
    print("backend using {}".format(backend))
    th.distributed.init_process_group(
        backend=backend,
        init_method=dist_init_method,
        world_size=world_size,
        rank=dev_id,
    )
    
    time.sleep(3)
    if proc_id == 0:
        queue.put(np.array([0.1, 0.2, 0.1, 0.2, 0.5, 0.2]))


def track_time():
    devices = [0, 1, 2, 3]
    n_gpus = len(devices)
    n_cpus = mp.cpu_count()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []

    for proc_id in range(n_gpus):
        p = ctx.Process(
            target=run,
            args=(
                proc_id,
                n_gpus,
                n_cpus // n_gpus,
                devices,
                queue,
            ),
        )
        p.start()

        procs.append(p)
    for p in procs:
        p.join()
    time_records = queue.get(block=False)
    num_exclude = 10  # exclude first 10 iterations
    if len(time_records) < 15:
        # exclude less if less records
        num_exclude = int(len(time_records) * 0.3)
    return np.mean(time_records[num_exclude:])
