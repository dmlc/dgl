"""This file defines the unified tensor framework interface required by DGL.

The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by DGL system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.

It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enbaled`` function
that returns whether the interface is supported by the framework or not.
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

    This function will be called only *once* during the initialization fo the
    backend module. The returned dictionary will become the attributes of the
    backend module.

    Examples
    --------
    >>> import torch as th
    >>> def data_type_dict():
    >>>   return { 'float16' : th.float16, 'float32' : th.float32, ... }

    After the module is initialized.

    >>> import backend as F
    >>> F.float16  # this will point to torch.float16

    Returns
    -------
    dict of str to data type
        The data type dict.
    """
    pass

def cpu():
    """Return a context object for CPU device."""
    pass

def tensor(data, dtype=None):
    """Create a tensor given the data and data type.

    Parameters
    ----------
    data : input data
        The interface should at least support list and numpy array.
        The data is copied to a newly-allocated tensor.
    dtype : data type, optional
        It should be one of the values in the data type dict.
        If is none, the type should be inferred from data.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def sparse_matrix(data, index, shape, force_format=False):
    """Create a sparse matrix.

    NOTE: Please make sure that the data and index tensors are not
    copied. This is critical to the performance.

    Parameters
    ----------
    data : Tensor
        Data tensor. It should be of shape (nnz,).
    index : tuple
        This is used to support different sparse formats.
        For COO format:
          index=('coo', coord), where coord is of shape (2, nnz).
          coord[0,:] should be the row index and coord[1,:] should be
          the column index.
        For CSR format:
          index=('csr', indices, indptr), where indices is of shape (nnz,)
          and indptr is of shape (nrows+1,). See ``scipy.sparse.csr_matrix``
          for more documents on what each array means.
    shape : tuple of int
        The shape.
    force_format : bool
        If true, the returned sparse matrix must be stored in the same
        format as the given index.

    Returns
    -------
    SparseMatrix
        The framework-specific sparse matrix. It can be stored in any format
        unless force_format is True.
    Tensor
        The data convert index due to sparse format change.
        None if no conversion is needed.
    """
    pass

def sparse_matrix_indices(spmat):
    """Return the indices of the given sparse matrix.

    Parameters
    ----------
    spmat : SparseMatrix
        The framework-specific sparse matrix.

    Returns
    -------
    index : tuple
        This is used to support different sparse formats.
        For COO format:
          index=('coo', coord), where coord is of shape (2, nnz).
          coord[0,:] should be the row index and coord[1,:] should be
          the column index.
        For CSR format:
          index=('csr', indices, indptr), where indices is of shape (nnz,)
          and indptr is of shape (nrows+1,). See ``scipy.sparse.csr_matrix``
          for more documents on what each array means.
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

def ndim(input):
    """Return the number of dimensions of the tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    int
        The number of dimensions
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

def mean(input, dim):
    """Reduce average the input tensor along the given dim.

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

def stack(seq, dim):
    """Stack the sequence of tensors along the given dimension.

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

def narrow_row(x, start, stop):
    """Narrow down the tensor along the first dimension.
    
    Parameters
    ----------
    x : Tensor
        The input tensor.
    start : int
        The start index (inclusive).
    stop : int
        The stop index (exclusive).

    Returns
    -------
    Tensor
        The narrowed tensor

    Notes
    -----
    The returned tensor could be a view of the original tensor.
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

def zeros(shape, dtype, ctx):
    """Create a zero tensor.

    Parameters
    ----------
    shape : tuple of int
        The tensor shape.
    dtype : data type
        It should be one of the values in the data type dict.
    ctx : context
        The device of the result tensor.

    Returns
    -------
    Tensor
        The zero tensor.
    """
    pass

def ones(shape, dtype, ctx):
    """Create a one tensor.

    Parameters
    ----------
    shape : tuple of int
        The tensor shape.
    dtype : data type
        It should be one of the values in the data type dict.
    ctx : context
        The device of the result tensor.

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

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    """Computes the sum along segments of a tensor.

    Equivalent to tf.unsorted_segment_sum, but seg_id is required to be a
    1D tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor
    seg_id : 1D Tensor
        The segment IDs whose values are between 0 and n_segs - 1.  Should
        have the same length as input.
    n_segs : int
        Number of distinct segments
    dim : int
        Dimension to sum on

    Returns
    -------
    Tensor
        The result
    """
    pass

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    """Computes the mean along segments of a tensor.

    Equivalent to tf.unsorted_segment_mean, but seg_id is required to be a
    1D tensor.

    Note that segments never appeared in seg_id will have results of 0.

    Parameters
    ----------
    input : Tensor
        The input tensor
    seg_id : 1D Tensor
        The segment IDs whose values are between 0 and n_segs - 1.  Should
        have the same length as input.
    n_segs : int
        Number of distinct segments
    dim : int
        Dimension to average on

    Returns
    -------
    Tensor
        The result
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

def rand_shuffle(arr):
    """Random shuffle the data in the first dimension of the array.

    The shuffled data is stored in a new array.

    Parameters
    ----------
    arr : Tensor
        The data tensor

    Returns
    -------
    Tensor
        The result tensor
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
