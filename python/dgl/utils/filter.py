from .._ffi.function import _init_api
from .. import backend as F


class Filter(object):
    """Class used to either find either find the subset of ids that are in this
    filter, or th subset of ids that are not in this filter,
    given a second set of ids.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.utils import Filter
    >>> f = Filter(th.tensor([3,2,9], device=th.device('cuda')))
    >>> f.find_included_indices(th.tensor([0,2,8,9], device=th.device('cuda')))
    tensor([1,3])
    >>> f.find_excluded_indices(th.tensor([0,2,8,9], device=th.device('cuda')))
    tensor([0,2], device='cuda')
    """
    def __init__(self, ids):
        """Create a new filter from a given set of ids. This currently is only
        implemented for the GPU.

        Parameters
        ----------
        ids : IdArray
            The unique set of ids to keep in the filter.
        """
        self._filter = _CAPI_DGLFilterCreateFromSet(
            F.zerocopy_to_dgl_ndarray(ids)) 

    def find_included_indices(self, test):
        """Find the index of the ids in `test` that are in this filter.

        Parameters
        ----------
        test : IdArray
            The set of ids to to test with.

        Returns
        -------
        IdArray
            The index of ids in `test` that are also in this filter.
        """
        return F.zerocopy_from_dgl_ndarray( \
            _CAPI_DGLFilterFindIncludedIndices( \
                self._filter, F.zerocopy_to_dgl_ndarray(test)))

    def find_excluded_indices(self, test):
        """Find the index of the ids in `test` that are not in this filter.

        Parameters
        ----------
        test : IdArray
            The set of ids to to test with.

        Returns
        -------
        IdArray
            The index of ids in `test` that are not in this filter.
        """
        return F.zerocopy_from_dgl_ndarray( \
            _CAPI_DGLFilterFindExcludedIndices( \
                self._filter, F.zerocopy_to_dgl_ndarray(test)))

_init_api("dgl.utils.filter")
