"""This file defines the unified tensor framework interface required by DGL.

The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by DGL system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.

It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
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

def as_scalar(data):
    """Returns a scalar whose value is copied from this array.

    Parameters
    ----------
    data : Tensor
        The input data

    Returns
    -------
    scalar
        The scalar value in the tensor.
    """
    pass

def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.

    Returns
    -------
    string
        the name of the preferred sparse matrix format.
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

def device_type(ctx):
    """Return a str representing device type"""
    pass

def device_id(ctx):
    """Return device index"""
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

def sum(input, dim, keepdims=False):
    """Reduce sum the input tensor along the given dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The reduce dim.
    keepdims : bool
        Whether to keep the summed dimension.

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """
    pass

def reduce_sum(input):
    """Returns the sum of all elements in the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A framework-specific tensor with shape (1,)
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

def reduce_mean(input):
    """Returns the average of all elements in the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A framework-specific tensor with shape (1,)
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

def reduce_max(input):
    """Returns the max of all elements in the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A framework-specific tensor with shape (1,)
    """
    pass

def min(input, dim):
    """Reduce min the input tensor along the given dim.

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

def reduce_min(input):
    """Returns the min of all elements in the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A framework-specific tensor with shape (1,)
    """
    pass


def argsort(input, dim, descending):
    """Return the indices that would sort the input along the given dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dim to sort along.
    descending : bool
        Controls the sorting order (False: ascending, True: descending)

    Returns
    -------
    Tensor
        A framework-specific tensor.
    """

def topk(input, k, dim, descending=True):
    """Return the k largest elements of the given input tensor along the given dimension.

    If descending is False then the k smallest elements are returned.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    k : int
        The number of elements.
    dim : int
        The dim to sort along.
    descending : bool
        Controls whether to return largest/smallest elements.
    """
    pass

def argtopk(input, k, dim, descending=True):
    """Return the indices of the k largest elements of the given input tensor
    along the given dimension.

    If descending is False then the k smallest elements are returned.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    k : int
        The number of elements.
    dim : int
        The dimension to sort along.
    descending : bool
        Controls whether to return largest/smallest elements.
    """
    pass

def exp(input):
    """Returns a new tensor with the exponential of the elements of the input tensor `input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor.
    """
    pass

def softmax(input, dim=-1):
    """Apply the softmax function on given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension along which to compute softmax.

    Returns
    -------
    Tensor
        The output tensor.
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

def repeat(input, repeats, dim):
    """Repeats elements of an array.

    Parameters
    ----------
    input : Tensor
        Input data array
    repeats : int
        The number of repetitions for each element
    dim : int
        The dim along which to repeat values.

    Returns
    -------
    Tensor
        The obtained tensor.
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

def slice_axis(data, axis, begin, end):
    """Slice along a given axis.
    Returns an array slice along a given axis starting from :attr:`begin` index to :attr:`end` index.

    Parameters
    ----------
    data : Tensor
        The data tensor.
    axis : int
        The axis along to slice the tensor.
    begin : int
        Indicates the begin index.
    end : int
        Indicates the end index.
    Returns:
    --------
    Tensor
        The sliced tensor.
    """
    pass

def take(data, indices, dim):
    """Takes elements from an input array along the given dim.

    Parameters
    ----------
    data : Tensor
        The data tensor.
    indices : Tensor
        The indices tensor.
    dim : Tensor
        The dimension to gather along.
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
    """Write the value into the data tensor using the row index inplace.

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

def swapaxes(input, axis1, axis2):
    """Interchange the two given axes of a tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axis1, axis2 : int
        The two axes.

    Returns
    -------
    Tensor
        The transposed tensor.
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

def zeros_like(input):
    """Create a zero tensor with the same shape, dtype and context of the
    given tensor.

    Parameters
    ----------
    input : Tensor
        The input

    Returns
    -------
    Tensor
        The result
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

def uniform(shape, dtype, ctx, low, high):
    """Crear a tensor with random value in an uniform 
    distribution between low (inclusive) and high (exclusive).

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
        The random tensor.
    """
    pass

def pad_packed_tensor(input, lengths, value, l_min=None):
    """Pads a packed batch of variable length tensors with given value.

    Parameters
    ----------
    input : Tensor
        The input tensor with shape :math:`(N, *)`
    lengths : list or tensor
        The array of tensor lengths (of the first dimension) :math:`L`.
        It should satisfy :math:`\sum_{i=1}^{B}L_i = N`,
        where :math:`B` is the length of :math:`L`.
    value : float
        The value to fill in the tensor.
    l_min : int or None, defaults to None.
        The minimum length each tensor need to be padded to, if set to None,
        then there is no minimum length requirement.

    Returns
    -------
    Tensor
        The obtained tensor with shape :math:`(B, \max(\max_i(L_i), l_{min}), *)`
    """
    pass

def pack_padded_tensor(input, lengths):
    """Packs a tensor containing padded sequence of variable length.

    Parameters
    ----------
    input : Tensor
        The input tensor with shape :math:`(B, L, *)`, where :math:`B` is
        the batch size and :math:`L` is the maximum length of the batch.
    lengths : list or tensor
        The array of tensor lengths (of the first dimension) :math:`L`.
        :math:`\max_i(L_i)` should equal :math:`L`.

    Returns
    -------
    Tensor
        The obtained tensor with shape :math:`(N, *)` where
        :math:`N = \sum_{i=1}^{B}L_i`
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

def boolean_mask(input, mask):
    """Selects elements in x according to the given mask from the first
    dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor
    mask : Boolean Tensor
        The mask

    Returns
    -------
    Tensor
        The result
    """
    pass

def equal(x, y):
    """Compares whether the elements are equal.

    Parameters
    ----------
    x, y : Tensor
        The two tensors

    Returns
    -------
    Boolean tensor
        The result, with the same shape as input.
    """
    pass

def logical_not(input):
    """Perform a logical not operation.  Equivalent to np.logical_not

    Parameters
    ----------
    input : Tensor
        The input

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

def full_1d(length, fill_value, dtype, ctx):
    """Create a 1D tensor full of the fill_value.

    Parameters
    ----------
    shape : int
        The length of the vector.
    fill_value : int
        The filled value.
    dtype : data type
        It should be one of the values in the data type dict.
    ctx : context
        The device of the result tensor.

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

def zerocopy_to_dgl_ndarray(input):
    """Zerocopy a framework-specific Tensor to dgl.ndarray.NDArray

    Parameters
    ----------
    input : Tensor

    Returns
    -------
    dgl.ndarray.NDArray
    """
    pass

def zerocopy_from_dgl_ndarray(input):
    """Zerocopy a dgl.ndarray.NDArray to framework-specific Tensor

    Parameters
    ----------
    input : dgl.ndarray.NDArray

    Returns
    -------
    Tensor
    """
    pass



###############################################################################
# Custom Operators for graph level computations.

# Note: These operators are supposed to be implemented using DGL-provided
# kernels (see kernel.py), and plug into tensor framework using custom op
# extensions.

def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map, rhs_map, out_map):
    """Perform binary operation between given data and reduce based on graph
    structure.

    Parameters
    ----------
    reducer : str
        Type of reduction: 'sum', 'max', 'min', 'mean', 'prod', 'none' (no
        reduction)
    binary_op : str
        Binary operation to perform, can be 'add', 'mul', 'sub', 'div'
    graph : GraphIndex
        The graph
    lhs : int
        The lhs target (src, dst, edge)
    rhs : int
        The rhs target (src, dst, edge)
    lhs_data : Tensor
        The lhs data
    rhs_data : Tensor
        The rhs data
    out_size : int
        Size of first dimension of output data
    lhs_map : tuple
        Two lhs id mapping arrays, one for forward pass, the other for backward
    rhs_map : tuple
        Two rhs id mapping arrays, one for forward pass, the other for backward
    out_map : tuple
        Two out id mapping arrays, one for forward pass, the other for backward

    Returns
    -------
    Tensor
        The result.
    """
    pass

def copy_reduce(reducer, graph, target, in_data, out_size, in_map, out_map):
    """Copy target data and perform reduce based on graph structure.

    Parameters
    ----------
    reducer : str
        Type of reduction: be 'sum', 'max', 'min', 'mean', 'prod', 'none' (no
        reduction)
    graph : GraphIndex
        The graph
    target : int
        The input target (src, dst, edge)
    in_data : Tensor
        The input data
    out_size : int
        Size of first dimension of output data
    in_map : tuple
        Two input id mapping arrays, one for forward, the other for backward
    out_map : tuple
        Two output id mapping arrays, one for forward, the other for backward

    Returns
    -------
    Tensor
        The result.
    """
    pass

###############################################################################
# Other interfaces
# ----------------
# These are not related to tensors. Some of them are temporary workarounds that
# should be included in DGL in the future.

def sync():
    """Synchronize computation.

    In DL frameworks such as MXNet and TensorFlow, the computation in operators
    are done asynchronously. This is to synchronize computation and makes sure
    that all computation is complete after this function call.
    """
    pass
