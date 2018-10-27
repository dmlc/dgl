"""This file defines the unified tensor framework interface required by DGL.

The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by DGL system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.
"""

###############################################################################
# Tensor, data type and context interfaces

def data_type_dict():
    """Returns a dictionary from data type string to the data type.

    The dictionary should include at least:
    float16
    float32
    float64
    uint8
    int8
    int16
    int32
    int64

    Examples
    --------
    >>> import torch as th
    >>> { 'float16' : th.float16, 'float32' : th.float32, ... }

    Returns
    -------
    dict of str to data type
        The data type dict.
    """
    pass

def context_dict():
    pass

def tensor(data, dtype=None):
    """Create a tensor given the data and data type.

    Parameters
    ----------
    data : input data
        The interface should at least support list and numpy array.
        The data is copied to a newly-allocated tensor.
    dtype : data type
        It should be one of the values in the data type dict.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def coo_tensor(idx, dat, shape):
    """Create a sparse tensor in COO format.

    Parameters
    ----------
    idx : Tensor
        Coordinate tensor. It should be of shape (2, nnz)
    dat : Tensor
        Data tensor. It should be of shape (nnz,)
    shape : tuple of int
        The shape of this tensor.

    Returns
    -------
    SparseTensor
        A framework-specific sparse tensor.
    """
    pass

def csr_tensor(data, indices, indptr, shape):
    """Create a sparse tensor in CSR format.

    See ``scipy.sparse.csr_matrix`` for more documents on what
    each argument means.

    Parameters
    ----------
    data : Tensor
        The data tensor. Should be of shape (nnz,)
    indices : Tensor
        The indices tensor. Should be of shape (nnz,)
    indptr : Tensor
        The indptr tensor. Should be of shape (nrows+1,)

    Returns
    -------
    SparseTensor
        A framework-specific sparse tensor.
    """
    pass

def is_tensor(obj):
    """Returns true if the given object is a framework-specific tensor."""
    pass

def shape(input):
    """Return the shape of the tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    tuple of int
        The tensor shape.
    """
    pass

def dtype(input):
    """Return the data type of the tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    data type
        It should be one of the values in the data type dict.
    """
    pass

def context(input):
    """Return the context/device of the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Context object
        A framework-specific context object.
    """
    pass

def astype(input, ty):
    """Convert the input tensor to the given data type.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    ty : data type
        It should be one of the values in the data type dict.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def asnumpy(input):
    """Convert the input tensor to numpy array.

    The data is copied.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    numpy.ndarray
        Numpy array.
    """
    pass

def copy_to(input, ctx):
    """Copy the given tensor to the context.

    Parameters
    ----------
    input : Tensor
        The input tensor
    ctx :
        A framework-specific context object.

    Returns
    -------
    Tensor
        The tensor on the given context.
    """
    pass

###############################################################################
# Tensor functions on feature data
# --------------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.

def sum(input, dim):
    """Reduce sum the input tensor along the given dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The reduce dim.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def max(input, dim):
    """Reduce max the input tensor along the given dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The reduce dim.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def cat(seq, dim):
    """Concat the sequence of tensors in the given dimension.

    Parameters
    ----------
    seq : list of Tensor
        The tensor sequence.
    dim : int
        The concat dim.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def split(input, sizes_or_sections, dim):
    """Split the input tensor into chunks.

    If ``sizes_or_sections`` is an integer, then the tensor will
    be splitted into equal pieces.

    If ``sizes_or_sections`` is a list, then the tensor will be
    splitted into segments.

    Parameters
    ----------
    input : Tensor

    Returns
    -------
    list of Tensor
        The splitted tensors.
    """
    pass

def gather_row(data, row_index):
    """Slice out the data given the row index.

    Parameters
    ----------
    data : Tensor
        The data tensor
    row_index : Tensor
        A 1-D integer tensor containing which rows to be sliced out.

    Returns
    -------
    Tensor
        The sliced data. The first dimension should equal to ``len(row_index)``.
    """
    pass

def scatter_row(data, row_index, value):
    """Write the value into the data tensor using the row index.

    This is an out-place write so it can work with autograd.

    Parameters
    ----------
    data : Tensor
        The data tensor to be updated.
    row_index : Tensor
        A 1-D integer tensor containing which rows to be updated.
    value : Tensor
        The new value.

    Returns
    -------
    Tensor
        The new data.
    """
    pass

def scatter_row_inplace(data, row_index, value):
    """Write the value into the data tensor using the row index inplacely.

    This is an inplace write so it will break the autograd.

    Parameters
    ----------
    data : Tensor
        The data tensor to be updated.
    row_index : Tensor
        A 1-D integer tensor containing which rows to be updated.
    value : Tensor
        The new value.

    Returns
    -------
    Tensor
        The new data.
    """
    pass

def squeeze(input, dim):
    """Remove the given dimension of size 1.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to be squeezed.

    Returns
    -------
    Tensor
        The result tensor.
    """
    pass

def unsqueeze(input, dim):
    """Add the given dimension of size 1.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to be unsqueezed.

    Returns
    -------
    Tensor
        The result tensor.
    """
    pass

def reshape(input, shape):
    """Reshape the tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    shape : tuple of int
        The new shape.

    Returns
    -------
    Tensor
        The reshaped tensor.
    """
    pass

def zeros(shape, dtype):
    """Create a zero tensor.

    Parameters
    ----------
    shape : tuple of int
        The tensor shape.
    dtype : data type
        It should be one of the values in the data type dict.

    Returns
    -------
    Tensor
        The zero tensor.
    """
    pass

def ones(shape, dtype):
    """Create a one tensor.

    Parameters
    ----------
    shape : tuple of int
        The tensor shape.
    dtype : data type
        It should be one of the values in the data type dict.

    Returns
    -------
    Tensor
        The one tensor.
    """
    pass

def spmm(x, y):
    """Multiply a sparse matrix with a dense matrix.

    Parameters
    ----------
    x : SparseTensor
        The sparse matrix.
    y : Tensor
        The dense matrix.

    Returns
    -------
    Tensor
        The result dense matrix.
    """
    pass

###############################################################################
# Tensor functions used *only* on index tensor
# ----------------
# These operators are light-weighted, so it is acceptable to fallback to
# numpy operators if currently missing in the framework. Ideally in the future,
# DGL should contain all the operations on index, so this set of operators
# should be gradually removed.

def unique(input):
    """Returns the unique scalar elements in a tensor.

    Parameters
    ----------
    input : Tensor
        Must be a 1-D tensor.

    Returns
    -------
    Tensor
        A 1-D tensor containing unique elements.
    """
    pass

def full_1d(length, fill_value):
    """Create a 1D tensor full of the fill_value.

    Parameters
    ----------
    shape : int
        The length of the vector.
    fill_value : int
        The filled value.

    Returns
    -------
    Tensor
        A result 1D tensor
    """
    pass

def nonzero_1d(input):
    """Return the nonzero index of the given 1D input.

    Parameters
    ----------
    input : Tensor
        Must be a 1D tensor.

    Returns
    -------
    Tensor
        A 1D integer tensor containing the nonzero indices.
    """
    pass

def sort_1d(input):
    """Sort a 1D tensor (in ascending order) and also return the original index.

    Parameters
    ----------
    input : Tensor
        The tensor to be sorted.

    Returns
    -------
    Tensor
        Sorted tensor.
    Tensor
        Index tensor of the elements in the original input.
    """
    pass

def arange(start, stop):
    """Create a 1D range int64 tensor.

    Parameters
    ----------
    start : int
        The range start.
    stop : int
        The range stop.

    Returns
    -------
    Tensor
        The result tensor.
    """
    pass

def zerocopy_to_dlpack(input):
    """Create a dlpack tensor that shares the input memory.

    Parameters
    ----------
    input : Tensor
        The input tensor

    Returns
    -------
    dlpack capsule
        A dlpack capsule that can be used by other framework.
    """
    pass

def zerocopy_from_dlpack(dlpack_tensor):
    """Create a tensor that shares the dlpack_tensor.

    Parameters
    ----------
    dlpack_tensor : dlpack capsule
        The dlpack tensor.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def zerocopy_to_numpy(input):
    """Create a numpy ndarray that shares the input memory.

    Parameters
    ----------
    input : Tensor
        The input tensor

    Returns
    -------
    numpy.ndarray
        A numpy ndarray.
    """
    pass

def zerocopy_from_numpy(np_array):
    """Create a tensor that shares the numpy array.

    Parameters
    ----------
    np_array : numpy.ndarray
        The numpy ndarray.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

###############################################################################
# Other interfaces
# ----------------
# These are not related to tensors. Some of them are temporary workarounds that
# should be included in DGL in the future.

def create_immutable_graph_index():
    """Create an immutable graph index object."""
    pass
