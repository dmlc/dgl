from time import time

import dgl


def get_partition_list(g, psize):
    tmp_time = time()
    print("getting adj using time{:.4f}".format(time() - tmp_time))
    print("run metis with partition size {}".format(psize))
    ret = dgl.transform.metis_partition(g, psize)
    print("metis finished in {} seconds.".format(time() - tmp_time))
    return list(ret.values())
