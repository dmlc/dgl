from collections import OrderedDict
from collections import deque
import sys
import os
from cachetools import *

total = 0
class _LRUCache:
    def __init__(self, capacity: int):
        self.hit = 0
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.hit += 1
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value


#implemented S3-fifo-cache in https://dl.acm.org/doi/abs/10.1145/3600006.3613147
class _FIFOCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  
        self.freq = {}   
        self.S = OrderedDict()# Small queue
        self.M = OrderedDict()# Main queue
        self.G = OrderedDict()# ghost queue
        self.hit = 0
        self._evict = 0

    def read(self, x):
        if x in self.S or x in self.M:
            # Cache Hit
            self.hit += 1
            self.freq[x] = min(self.freq.get(x, 0) + 1, 3) 
            return x
        else:
            # Cache Miss
            self.insert(x)
            self.freq[x] = 0
            return -1

    def insert(self, x):
        while len(self.S) + len(self.M) >= self.capacity:
            self.evict()
        if x in self.G:
            self.M[x] = x
            self.M.move_to_end(x)
        else:
            self.S[x] = x
            self.S.move_to_end(x)

    def evict(self):
        self._evict+=1
        if len(self.S) >= 0.1 * self.capacity:
            self.evict_S()
        else:
            self.evict_M()

    def evict_S(self):
        evicted = False
        while not evicted and self.S:
            t = self.S.popitem(last=False)
            if self.freq.get(t, 0) > 1:
                self.M[t] = t
                self.M.move_to_end(t)
                if len(self.M) > self.capacity:
                    self.evict_M()
                else:
                    self.G[t] = t
                    self.G.move_to_end(t)
                evicted = True

    def evict_M(self):
        evicted = False
        while not evicted and self.M:
            t = self.M.popitem(last=False)
            if self.freq.get(t, 0) > 0:
                self.M[t] = t
                self.M.move_to_end(t)
            else:
                evicted = True





if __name__ == '__main__':
    cache = FIFOCache(int(16777216*4/1.7))
    hit = 0
    import time
    for line in sys.stdin:
        line = line[:-1]
        line = line.split(',')
        time1 = 0
        time2 = 0
        for it in line:
            st_time = time.time()
            if ((int(it)) in cache) :
                hit += 1
            else:
                cache.__setitem__(int(it),int(it))
            end_time = time.time()
            total += 1
            time1 += (end_time-st_time)
        print(float(hit)/total)
    print(str(time1))
    print(float(hit)/total)

    # cache2 = FIFOCache(16777216)
    # for line in sys.stdin:
    #     line = line[:-1]
    #     line = line.split(',')
    #     for it in line:
    #         cache2.read(int(int(it) * 1.7 / 4))
    #         total += 1
    # print(str(float(cache2.hit)/total) + '\t' + str(cache2.hit) + '\t' + str(total))

