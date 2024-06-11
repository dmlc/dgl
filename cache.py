from collections import OrderedDict
from collections import deque
import sys
import os
#from cachetools import *
from cachemonCache import S3FIFO

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





if __name__ == '__main__':
    S3FIFOcache = S3FIFO(cache_size=int(16777216/1.7)) #16GB   
    hit = 0
    import time
    for line in sys.stdin:
        line = line[:-1]
        line = line.split(',')
        time1 = 0
        time2 = 0
        for it in line:
            st_time = time.time()
            if (int(it) in S3FIFOcache) :
                hit += 1
            else:
                S3FIFOcache.put(int(it),int(it))
                #cache.__setitem__(int(it),int(it))
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

