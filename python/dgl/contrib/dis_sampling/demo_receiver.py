# This is a demo code shows that how to implement a receiver
import mxnet as mx
import numpy as np

import receiver
import time

_LOCAL_PORT = 50051

def start_receiver():
    my_receiver = receiver.Receiver()
    my_receiver.start(_LOCAL_PORT)
    while True:
        sub_graph = my_receiver.recv()
        print(sub_graph)
        time.sleep(1)

if __name__ == '__main__':
    start_receiver()