""" API for managing array partitionings """

from .base import DGLError
from ._ffi.function import _init_api


class NDArrayPartition(object):
    """ Create a new partition of an NDArray. That is, an object which assigns
    each row of an NDArray to a specific partition.

    Parameters
    ----------
    array_size : int
        The first dimension of the array being partitioned.
    num_parts : int
        The number of parts to divide the array into.
    mode : String
        The type of partition. Currently, the only valid value is 'remainder',
        which assigns rows based on remainder when dividing the row id by the
        number of parts (e.g., i % num_parts).
    part_ranges : List
        Currently unused.
    """
    def __init__(self, array_size, num_parts, mode='remainder', part_ranges=None):
        assert num_parts > 0, 'Invalid "num_parts", must be > 0.'
        if mode == 'range':
            assert part_ranges not is None, 'When using a range-based ' \
                    'partitioning, the range must be provided.'
            assert part_ranges[-1] == array_size, '"part_ranges" must cover ' \
                    'the entire array.'
            assert len(part_ranges) == num_parts + 1, 'The size of ' \
                    '"part_ranges" must be equal to "num_parts+1"'
            raise DGLError('Range based partitions are not yet supported.')
        elif mode == 'remainder':
            assert part_ranges is None, 'When using remainder-based ' \
                    'partitioning, "part_ranges" should not be specified.'
            self._partition = _CAPI_DGLNDArrayPartitionCreateRemainderBased(
                    array_size, num_parts)
        else:
            assert False, 'Unknown partition mode "{}"'.format(mode)


    def get(self):
        return self._partition


_init_api("dgl.partition")
