""" API for managing array partitionings """

from ..base import DGLError

_MODES_MAP = {
    'remainder': 0
    'range': 1
}

MODES = list(_MODES_MAP.keys())


class NDArrayPartition(object):
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
