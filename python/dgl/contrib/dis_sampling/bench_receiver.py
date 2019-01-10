# Benchmark receiver
import mxnet as mx
import numpy as np
import receiver
import time

_LOCAL_PORT = 2049

_ITER = 1000

def start_receiver():
    my_receiver = receiver.Receiver(port=_LOCAL_PORT, buffer_size=20)
    server = my_receiver.start()
    sub_graph = my_receiver.recv()
    print("recv!")
    start = time.time()
    for n in range(_ITER):
        sub_graph = my_receiver.recv()
        print(n)
    elapsed = (time.time() - start)    
    print("time: " + str(elapsed))

if __name__ == '__main__':
    start_receiver()
    while True:
        time.sleep(1)