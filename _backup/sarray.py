# -*- coding: utf-8 -*-
# Copyright © 2017 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
'''
This module defines the SArray class which provides the
ability to create, access and manipulate a remote scalable array object.

SArray acts similarly to pandas.Series but without indexing.
The data is immutable, homogeneous, and is stored on the Turi Server side.
'''
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from ..connect import main as glconnect
from ..cython.cy_flexible_type import pytype_from_dtype, pytype_from_array_typecode
from ..cython.cy_flexible_type import infer_type_of_list, infer_type_of_sequence
from ..cython.cy_sarray import UnitySArrayProxy
from ..cython.context import debug_trace as cython_context
from ..util import _is_non_string_iterable, _make_internal_url
from ..visualization import _get_client_app_path
from ..visualization import Plot
from .image import Image as _Image
from .. import aggregate as _aggregate
from ..deps import numpy, HAS_NUMPY
from ..deps import pandas, HAS_PANDAS

import time
import sys
import array
import collections
import datetime
import warnings
import numbers
import six

__all__ = ['SArray']

if sys.version_info.major > 2:
    long = int

def _create_sequential_sarray(size, start=0, reverse=False):
    if type(size) is not int:
        raise TypeError("size must be int")

    if type(start) is not int:
        raise TypeError("size must be int")

    if type(reverse) is not bool:
        raise TypeError("reverse must me bool")

    with cython_context():
        return SArray(_proxy=glconnect.get_unity().create_sequential_sarray(size, start, reverse))

def load_sarray(filename):
    """
    Load an SArray. The filename extension is used to determine the format
    automatically. This function is particurly useful for SArrays previously
    saved in binary format. If the SArray is in binary format, ``filename`` is
    actually a directory, created when the SArray is saved.

    Paramaters
    ----------
    filename : string
        Location of the file to load. Can be a local path or a remote URL.

    Returns
    -------
    out : SArray

    See Also
    --------
    SArray.save

    Examples
    --------
    >>> sa = turicreate.SArray(data=[1,2,3,4,5])
    >>> sa.save('./my_sarray')
    >>> sa_loaded = turicreate.load_sarray('./my_sarray')
    """
    sa = SArray(data=filename)
    return sa


class SArray(object):
    """
    An immutable, homogeneously typed array object backed by persistent storage.

    SArray is scaled to hold data that are much larger than the machine's main
    memory. It fully supports missing values and random access. The
    data backing an SArray is located on the same machine as the Turi
    Server process. Each column in an :py:class:`~turicreate.SFrame` is an
    SArray.

    Parameters
    ----------
    data : list | numpy.ndarray | pandas.Series | string
        The input data. If this is a list, numpy.ndarray, or pandas.Series,
        the data in the list is converted and stored in an SArray.
        Alternatively if this is a string, it is interpreted as a path (or
        url) to a text file. Each line of the text file is loaded as a
        separate row. If ``data`` is a directory where an SArray was previously
        saved, this is loaded as an SArray read directly out of that
        directory.

    dtype : {None, int, float, str, list, array.array, dict, datetime.datetime, turicreate.Image}, optional
        The data type of the SArray. If not specified (None), we attempt to
        infer it from the input. If it is a numpy array or a Pandas series, the
        dtype of the array/series is used. If it is a list, the dtype is
        inferred from the inner list. If it is a URL or path to a text file, we
        default the dtype to str.

    ignore_cast_failure : bool, optional
        If True, ignores casting failures but warns when elements cannot be
        casted into the specified dtype.

    Notes
    -----
    - If ``data`` is pandas.Series, the index will be ignored.
    - The datetime is based on the Boost datetime format (see http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html
      for details)

    Examples
    --------
    SArray can be constructed in various ways:

    Construct an SArray from list.

    >>> from turicreate import SArray
    >>> sa = SArray(data=[1,2,3,4,5], dtype=int)

    Construct an SArray from numpy.ndarray.

    >>> sa = SArray(data=numpy.asarray([1,2,3,4,5]), dtype=int)
    or:
    >>> sa = SArray(numpy.asarray([1,2,3,4,5]), int)

    Construct an SArray from pandas.Series.

    >>> sa = SArray(data=pd.Series([1,2,3,4,5]), dtype=int)
    or:
    >>> sa = SArray(pd.Series([1,2,3,4,5]), int)

    If the type is not specified, automatic inference is attempted:

    >>> SArray(data=[1,2,3,4,5]).dtype
    int
    >>> SArray(data=[1,2,3,4,5.0]).dtype
    float

    The SArray supports standard datatypes such as: integer, float and string.
    It also supports three higher level datatypes: float arrays, dict
    and list (array of arbitrary types).

    Create an SArray from a list of strings:

    >>> sa = SArray(data=['a','b'])

    Create an SArray from a list of float arrays;

    >>> sa = SArray([[1,2,3], [3,4,5]])

    Create an SArray from a list of lists:

    >>> sa = SArray(data=[['a', 1, {'work': 3}], [2, 2.0]])

    Create an SArray from a list of dictionaries:

    >>> sa = SArray(data=[{'a':1, 'b': 2}, {'b':2, 'c': 1}])

    Create an SArray from a list of datetime objects:

    >>> sa = SArray(data=[datetime.datetime(2011, 10, 20, 9, 30, 10)])

    Construct an SArray from local text file. (Only works for local server).

    >>> sa = SArray('/tmp/a_to_z.txt.gz')

    Construct an SArray from a text file downloaded from a URL.

    >>> sa = SArray('http://s3-us-west-2.amazonaws.com/testdatasets/a_to_z.txt.gz')

    **Numeric Operators**

    SArrays support a large number of vectorized operations on numeric types.
    For instance:

    >>> sa = SArray([1,1,1,1,1])
    >>> sb = SArray([2,2,2,2,2])
    >>> sc = sa + sb
    >>> sc
    dtype: int
    Rows: 5
    [3, 3, 3, 3, 3]
    >>> sc + 2
    dtype: int
    Rows: 5
    [5, 5, 5, 5, 5]

    Operators which are supported include all numeric operators (+,-,*,/), as
    well as comparison operators (>, >=, <, <=), and logical operators (&, | ).

    For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> (sa >= 2) & (sa <= 4)
    dtype: int
    Rows: 5
    [0, 1, 1, 1, 0]

    The numeric operators (+,-,*,/) also work on array types:

    >>> sa = SArray(data=[[1.0,1.0], [2.0,2.0]])
    >>> sa + 1
    dtype: list
    Rows: 2
    [array('f', [2.0, 2.0]), array('f', [3.0, 3.0])]
    >>> sa + sa
    dtype: list
    Rows: 2
    [array('f', [2.0, 2.0]), array('f', [4.0, 4.0])]

    The addition operator (+) can also be used for string concatenation:

    >>> sa = SArray(data=['a','b'])
    >>> sa + "x"
    dtype: str
    Rows: 2
    ['ax', 'bx']

    This can be useful for performing type interpretation of lists or
    dictionaries stored as strings:

    >>> sa = SArray(data=['a,b','c,d'])
    >>> ("[" + sa + "]").astype(list) # adding brackets make it look like a list
    dtype: list
    Rows: 2
    [['a', 'b'], ['c', 'd']]

    All comparison operations and boolean operators are supported and emit
    binary SArrays.

    >>> sa = SArray([1,2,3,4,5])
    >>> sa >= 2
    dtype: int
    Rows: 3
    [0, 1, 1, 1, 1]
    >>> (sa >= 2) & (sa <= 4)
    dtype: int
    Rows: 3
    [0, 1, 1, 1, 0]


    **Element Access and Slicing**
    SArrays can be accessed by integer keys just like a regular python list.
    Such operations may not be fast on large datasets so looping over an SArray
    should be avoided.

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[0]
    1
    >>> sa[2]
    3
    >>> sa[5]
    IndexError: SFrame index out of range

    Negative indices can be used to access elements from the tail of the array

    >>> sa[-1] # returns the last element
    5
    >>> sa[-2] # returns the second to last element
    4

    The SArray also supports the full range of python slicing operators:

    >>> sa[1000:] # Returns an SArray containing rows 1000 to the end
    >>> sa[:1000] # Returns an SArray containing rows 0 to row 999 inclusive
    >>> sa[0:1000:2] # Returns an SArray containing rows 0 to row 1000 in steps of 2
    >>> sa[-100:] # Returns an SArray containing last 100 rows
    >>> sa[-100:len(sa):2] # Returns an SArray containing last 100 rows in steps of 2

    **Logical Filter**

    An SArray can be filtered using

    >>> array[binary_filter]

    where array and binary_filter are SArrays of the same length. The result is
    a new SArray which contains only elements of 'array' where its matching row
    in the binary_filter is non zero.

    This permits the use of boolean operators that can be used to perform
    logical filtering operations.  For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[(sa >= 2) & (sa <= 4)]
    dtype: int
    Rows: 3
    [2, 3, 4]

    This can also be used more generally to provide filtering capability which
    is otherwise not expressible with simple boolean functions. For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[sa.apply(lambda x: math.log(x) <= 1)]
    dtype: int
    Rows: 3
    [1, 2]

    This is equivalent to

    >>> sa.filter(lambda x: math.log(x) <= 1)
    dtype: int
    Rows: 3
    [1, 2]

    **Iteration**

    The SArray is also iterable, but not efficiently since this involves a
    streaming transmission of data from the server to the client. This should
    not be used for large data.

    >>> sa = SArray([1,2,3,4,5])
    >>> [i + 1 for i in sa]
    [2, 3, 4, 5, 6]

    This can be used to convert an SArray to a list:

    >>> sa = SArray([1,2,3,4,5])
    >>> l = list(sa)
    >>> l
    [1, 2, 3, 4, 5]
    """

    __slots__ = ["__proxy__", "_getitem_cache"]

    def __init__(self, data=[], dtype=None, ignore_cast_failure=False, _proxy=None):
        """
        __init__(data=list(), dtype=None, ignore_cast_failure=False)

        Construct a new SArray. The source of data includes: list,
        numpy.ndarray, pandas.Series, and urls.
        """

        if dtype is not None and type(dtype) != type:
            raise TypeError('dtype must be a type, e.g. use int rather than \'int\'')

        if (_proxy):
            self.__proxy__ = _proxy
        elif type(data) == SArray:
            self.__proxy__ = data.__proxy__
        else:
            self.__proxy__ = UnitySArrayProxy()
            # we need to perform type inference
            if dtype is None:
                if HAS_PANDAS and isinstance(data, pandas.Series):
                    # if it is a pandas series get the dtype of the series
                    dtype = pytype_from_dtype(data.dtype)
                    if dtype == object:
                        # we need to get a bit more fine grained than that
                        dtype = infer_type_of_sequence(data.values)

                elif HAS_NUMPY and isinstance(data, numpy.ndarray):
                    # first try the fast inproc method
                    try:
                        from .. import numpy_loader
                        if numpy_loader.numpy_activation_successful():
                            from ..numpy import _fast_numpy_to_sarray
                            ret = _fast_numpy_to_sarray(data)
                            # conversion is good!
                            # swap the proxy.
                            self.__proxy__, ret.__proxy__ = ret.__proxy__, self.__proxy__
                            return
                        else:
                            dtype = infer_type_of_sequence(data)
                    except:
                        pass

                    # if it is a numpy array, get the dtype of the array
                    dtype = pytype_from_dtype(data.dtype)
                    if dtype == object:
                        # we need to get a bit more fine grained than that
                        dtype = infer_type_of_sequence(data)
                    if len(data.shape) == 2:
                        # we need to make it an array or a list
                        if dtype == float or dtype == int:
                            dtype = array.array
                        else:
                            dtype = list
                    elif len(data.shape) > 2:
                        raise TypeError("Cannot convert Numpy arrays of greater than 2 dimensions")

                elif (isinstance(data, str) or
                      (sys.version_info.major < 3 and isinstance(data, unicode))):
                    # if it is a file, we default to string
                    dtype = str
                elif isinstance(data, array.array):
                    dtype = pytype_from_array_typecode(data.typecode)
                elif isinstance(data, collections.Sequence):
                    # Covers any ordered python container and arrays.
                    # Convert it to a list first.
                    dtype = infer_type_of_sequence(data)
                else:
                    dtype = None

            if HAS_PANDAS and isinstance(data, pandas.Series):
                with cython_context():
                    self.__proxy__.load_from_iterable(data.values, dtype, ignore_cast_failure)
            elif (isinstance(data, str) or (sys.version_info.major <= 2 and isinstance(data, unicode))):
                internal_url = _make_internal_url(data)
                with cython_context():
                    self.__proxy__.load_autodetect(internal_url, dtype)
            elif ((HAS_NUMPY and isinstance(data, numpy.ndarray))
                  or isinstance(data, array.array)
                  or isinstance(data, collections.Sequence)):

                with cython_context():
                    self.__proxy__.load_from_iterable(data, dtype, ignore_cast_failure)
            else:
                raise TypeError("Unexpected data source. " \
                                "Possible data source types are: list, " \
                                "numpy.ndarray, pandas.Series, and string(url)")

    @classmethod
    def date_range(cls,start_time,end_time,freq):
        '''
        Returns a new SArray that represents a fixed frequency datetime index.

        Parameters
        ----------
        start_time : datetime.datetime
          Left bound for generating dates.

        end_time : datetime.datetime
          Right bound for generating dates.

        freq : datetime.timedelta
          Fixed frequency between two consecutive data points.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import datetime as dt
        >>> start = dt.datetime(2013, 5, 7, 10, 4, 10)
        >>> end = dt.datetime(2013, 5, 10, 10, 4, 10)
        >>> sa = tc.SArray.date_range(start,end,dt.timedelta(1))
        >>> print sa
        dtype: datetime
        Rows: 4
        [datetime.datetime(2013, 5, 7, 10, 4, 10),
         datetime.datetime(2013, 5, 8, 10, 4, 10),
         datetime.datetime(2013, 5, 9, 10, 4, 10),
         datetime.datetime(2013, 5, 10, 10, 4, 10)]
       '''

        if not isinstance(start_time,datetime.datetime):
            raise TypeError("The ``start_time`` argument must be from type datetime.datetime.")

        if not isinstance(end_time,datetime.datetime):
            raise TypeError("The ``end_time`` argument must be from type datetime.datetime.")

        if not isinstance(freq,datetime.timedelta):
            raise TypeError("The ``freq`` argument must be from type datetime.timedelta.")

        from .. import extensions
        return extensions.date_range(start_time,end_time,freq.total_seconds())

    @classmethod
    def from_const(cls, value, size, dtype=type(None)):
        """
        Constructs an SArray of size with a const value.

        Parameters
        ----------
        value : [int | float | str | array.array | list | dict | datetime]
          The value to fill the SArray
        size : int
          The size of the SArray
        dtype : type
          The type of the SArray. If not specified, is automatically detected
          from the value. This should be specified if value=None since the
          actual type of the SArray can be anything.

        Examples
        --------
        Construct an SArray consisting of 10 zeroes:

        >>> turicreate.SArray.from_const(0, 10)

        Construct an SArray consisting of 10 missing string values:

        >>> turicreate.SArray.from_const(None, 10, str)
        """
        assert isinstance(size, (int, long)) and size >= 0, "size must be a positive int"
        if not isinstance(value, (type(None), int, float, str, array.array, list, dict, datetime.datetime)):
            raise TypeError('Cannot create sarray of value type %s' % str(type(value)))
        proxy = UnitySArrayProxy()
        proxy.load_from_const(value, size, dtype)
        return cls(_proxy=proxy)

    @classmethod
    def from_sequence(cls, *args):
        """
        from_sequence(start=0, stop)

        Create an SArray from sequence

        .. sourcecode:: python

            Construct an SArray of integer values from 0 to 999

            >>> tc.SArray.from_sequence(1000)

            This is equivalent, but more efficient than:

            >>> tc.SArray(range(1000))

            Construct an SArray of integer values from 10 to 999

            >>> tc.SArray.from_sequence(10, 1000)

            This is equivalent, but more efficient than:

            >>> tc.SArray(range(10, 1000))

        Parameters
        ----------
        start : int, optional
            The start of the sequence. The sequence will contain this value.

        stop : int
          The end of the sequence. The sequence will not contain this value.

        """
        start = None
        stop = None
        # fill with args. This checks for from_sequence(100), from_sequence(10,100)
        if len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]

        if stop is None and start is None:
            raise TypeError("from_sequence expects at least 1 argument. got 0")
        elif start is None:
            return _create_sequential_sarray(stop)
        else:
            size = stop - start
            # this matches the behavior of range
            # i.e. range(100,10) just returns an empty array
            if (size < 0):
                size = 0
            return _create_sequential_sarray(size, start)

    @classmethod
    def read_json(cls, filename):
        """
        Construct an SArray from a json file or glob of json files.
        The json file must contain a list of dictionaries. The returned
        SArray type will be of dict type

        Parameters
        ----------
        filename : str
          The filename or glob to load into an SArray.

        Examples
        --------
        Construct an SArray from a local JSON file named 'data.json':

        >>> turicreate.SArray.read_json('/data/data.json')

        Construct an SArray from all JSON files /data/data*.json

        >>> turicreate.SArray.read_json('/data/data*.json')

        """
        proxy = UnitySArrayProxy()
        proxy.load_from_json_record_files(_make_internal_url(filename))
        return cls(_proxy = proxy)

    @classmethod
    def where(cls, condition, istrue, isfalse, dtype=None):
        """
        Selects elements from either istrue or isfalse depending on the value
        of the condition SArray.

        Parameters
        ----------
        condition : SArray
        An SArray of values such that for each value, if non-zero, yields a
        value from istrue, otherwise from isfalse.

        istrue : SArray or constant
        The elements selected if condition is true. If istrue is an SArray,
        this must be of the same length as condition.

        isfalse : SArray or constant
        The elements selected if condition is false. If istrue is an SArray,
        this must be of the same length as condition.

        dtype : type
        The type of result SArray. This is required if both istrue and isfalse
        are constants of ambiguous types.

        Examples
        --------

        Returns an SArray with the same values as g with values above 10
        clipped to 10

        >>> g = SArray([6,7,8,9,10,11,12,13])
        >>> SArray.where(g > 10, 10, g)
        dtype: int
        Rows: 8
        [6, 7, 8, 9, 10, 10, 10, 10]

        Returns an SArray with the same values as g with values below 10
        clipped to 10

        >>> SArray.where(g > 10, g, 10)
        dtype: int
        Rows: 8
        [10, 10, 10, 10, 10, 11, 12, 13]

        Returns an SArray with the same values of g with all values == 1
        replaced by None

        >>> g = SArray([1,2,3,4,1,2,3,4])
        >>> SArray.where(g == 1, None, g)
        dtype: int
        Rows: 8
        [None, 2, 3, 4, None, 2, 3, 4]

        Returns an SArray with the same values of g, but with each missing value
        replaced by its corresponding element in replace_none

        >>> g = SArray([1,2,None,None])
        >>> replace_none = SArray([3,3,2,2])
        >>> SArray.where(g != None, g, replace_none)
        dtype: int
        Rows: 4
        [1, 2, 2, 2]
        """
        true_is_sarray = isinstance(istrue, SArray)
        false_is_sarray = isinstance(isfalse, SArray)
        if not true_is_sarray and false_is_sarray:
            istrue = cls(_proxy=condition.__proxy__.to_const(istrue, isfalse.dtype))
        if true_is_sarray and not false_is_sarray:
            isfalse = cls(_proxy=condition.__proxy__.to_const(isfalse, istrue.dtype))
        if not true_is_sarray and not false_is_sarray:
            if dtype is None:
                if istrue is None:
                    dtype = type(isfalse)
                elif isfalse is None:
                    dtype = type(istrue)
                elif type(istrue) != type(isfalse):
                    raise TypeError("true and false inputs are of different types")
                elif type(istrue) == type(isfalse):
                    dtype = type(istrue)
            if dtype is None:
                raise TypeError("Both true and false are None. Resultant type cannot be inferred.")
            istrue = cls(_proxy=condition.__proxy__.to_const(istrue, dtype))
            isfalse = cls(_proxy=condition.__proxy__.to_const(isfalse, dtype))
        return cls(_proxy=condition.__proxy__.ternary_operator(istrue.__proxy__, isfalse.__proxy__))

    def to_numpy(self):
        """
        Converts this SArray to a numpy array

        This operation will construct a numpy array in memory. Care must
        be taken when size of the returned object is big.

        Returns
        -------
        out : numpy.ndarray
            A Numpy Array containing all the values of the SArray

        """
        assert HAS_NUMPY, 'numpy is not installed.'
        import numpy
        return numpy.asarray(self)

    def __get_content_identifier__(self):
        """
        Returns the unique identifier of the content that backs the SArray

        Notes
        -----
        Meant for internal use only.
        """
        with cython_context():
            return self.__proxy__.get_content_identifier()

    def save(self, filename, format=None):
        """
        Saves the SArray to file.

        The saved SArray will be in a directory named with the `targetfile`
        parameter.

        Parameters
        ----------
        filename : string
            A local path or a remote URL.  If format is 'text', it will be
            saved as a text file. If format is 'binary', a directory will be
            created at the location which will contain the SArray.

        format : {'binary', 'text', 'csv'}, optional
            Format in which to save the SFrame. Binary saved SArrays can be
            loaded much faster and without any format conversion losses.
            'text' and 'csv' are synonymous: Each SArray row will be written
            as a single line in an output text file. If not
            given, will try to infer the format from filename given. If file
            name ends with 'csv', 'txt' or '.csv.gz', then save as 'csv' format,
            otherwise save as 'binary' format.
        """
        from .sframe import SFrame as _SFrame

        if format is None:
            if filename.endswith(('.csv', '.csv.gz', 'txt')):
                format = 'text'
            else:
                format = 'binary'
        if format == 'binary':
            with cython_context():
                self.__proxy__.save(_make_internal_url(filename))
        elif format == 'text' or format == 'csv':
            sf = _SFrame({'X1':self})
            with cython_context():
                sf.__proxy__.save_as_csv(_make_internal_url(filename), {'header':False})
        else:
            raise ValueError("Unsupported format: {}".format(format))

    def __repr__(self):
        """
        Returns a string description of the SArray.
        """
        data_str = self.__str__()
        ret = "dtype: " + str(self.dtype.__name__) + "\n"
        if (self.__has_size__()):
            ret = ret + "Rows: " + str(len(self)) + "\n"
        else:
            ret = ret + "Rows: ?\n"
        ret = ret + data_str
        return ret

    def __str__(self):
        """
        Returns a string containing the first 100 elements of the array.
        """

        # If sarray is image, take head of elements casted to string.
        if self.dtype == _Image:
            headln = str(list(self.astype(str).head(100)))
        else:
            if sys.version_info.major < 3:
                headln = str(list(self.head(100)))
                headln = unicode(headln.decode('string_escape'),'utf-8',errors='replace').encode('utf-8')
            else:
                headln = str(list(self.head(100)))
        if (self.__proxy__.has_size() is False or len(self) > 100):
            # cut the last close bracket
            # and replace it with ...
            headln = headln[0:-1] + ", ... ]"
        return headln

    def __nonzero__(self):
        """
        Raises a ValueError exception.
        The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().
        """
        # message copied from Numpy
        raise ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")

    def __bool__(self):
        """
        Raises a ValueError exception.
        The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().
        """
        # message copied from Numpy
        raise ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")

    def __len__(self):
        """
        Returns the length of the array
        """
        return self.__proxy__.size()

    def __iter__(self):
        """
        Provides an iterator to the contents of the array.
        """
        def generator():
            elems_at_a_time = 262144
            self.__proxy__.begin_iterator()
            ret = self.__proxy__.iterator_get_next(elems_at_a_time)
            while(True):
                for j in ret:
                    yield j

                if len(ret) == elems_at_a_time:
                    ret = self.__proxy__.iterator_get_next(elems_at_a_time)
                else:
                    break

        return generator()

    def __contains__(self, item):
        """
        Returns true if any element in this SArray is identically equal to item.

        Following are equivalent:

        >>> element in sa
        >>> sa.__contains__(element)

        For an element-wise contains see ``SArray.contains``

        """
        return (self == item).any()

    @property
    def shape(self):
        """
        The shape of the SArray, in a tuple. The first entry is the number of
        rows.

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.shape
        (3,)
        """
        return (len(self),)

    def contains(self, item):
        """
        Performs an element-wise search of "item" in the SArray.

        Conceptually equivalent to:

        >>> sa.apply(lambda x: item in x)

        If the current SArray contains strings and item is a string. Produces a 1
        for each row if 'item' is a substring of the row and 0 otherwise.

        If the current SArray contains list or arrays, this produces a 1
        for each row if 'item' is an element of the list or array.

        If the current SArray contains dictionaries, this produces a 1
        for each row if 'item' is a key in the dictionary.

        Parameters
        ----------
        item : any type
            The item to search for.

        Returns
        -------
        out : SArray
            A binary SArray where a non-zero value denotes that the item
            was found in the row. And 0 if it is not found.

        Examples
        --------
        >>> SArray(['abc','def','ghi']).contains('a')
        dtype: int
        Rows: 3
        [1, 0, 0]
        >>> SArray([['a','b'],['b','c'],['c','d']]).contains('b')
        dtype: int
        Rows: 3
        [1, 1, 0]
        >>> SArray([{'a':1},{'a':2,'b':1}, {'c':1}]).contains('a')
        dtype: int
        Rows: 3
        [1, 1, 0]

        See Also
        --------
        is_in
        """
        return SArray(_proxy = self.__proxy__.left_scalar_operator(item, 'in'))


    def is_in(self, other):
        """
        Performs an element-wise search for each row in 'other'.

        Conceptually equivalent to:

        >>> sa.apply(lambda x: x in other)

        If the current SArray contains strings and other is a string. Produces a 1
        for each row if the row is a substring of 'other', and 0 otherwise.

        If the 'other' is a list or array, this produces a 1
        for each row if the row is an element of 'other'

        Parameters
        ----------
        other : list, array.array, str
            The variable to search in.

        Returns
        -------
        out : SArray
            A binary SArray where a non-zero value denotes that row was
            was found in 'other'. And 0 if it is not found.

        Examples
        --------
        >>> SArray(['ab','bc','cd']).is_in('abc')
        dtype: int
        Rows: 3
        [1, 1, 0]
        >>> SArray(['a','b','c']).is_in(['a','b'])
        dtype: int
        Rows: 3
        [1, 1, 0]

        See Also
        --------
        contains
        """
        return SArray(_proxy = self.__proxy__.right_scalar_operator(other, 'in'))

    # XXX: all of these functions are highly repetitive
    def __add__(self, other):
        """
        If other is a scalar value, adds it to the current array, returning
        the new result. If other is an SArray, performs an element-wise
        addition of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '+'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '+'))

    def __sub__(self, other):
        """
        If other is a scalar value, subtracts it from the current array, returning
        the new result. If other is an SArray, performs an element-wise
        subtraction of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '-'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '-'))

    def __mul__(self, other):
        """
        If other is a scalar value, multiplies it to the current array, returning
        the new result. If other is an SArray, performs an element-wise
        multiplication of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '*'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '*'))

    def __div__(self, other):
        """
        If other is a scalar value, divides each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise division of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '/'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '/'))

    def __truediv__(self, other):
        """
        If other is a scalar value, divides each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise division of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '/'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '/'))


    def __floordiv__(self, other):
        """
        If other is a scalar value, divides each element of the current array
        by the value, returning floor of the result. If other is an SArray, performs
        an element-wise division of the two arrays returning the floor of the result.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '//'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '//'))

    def __pow__(self, other):
        """
        If other is a scalar value, raises each element of the current array to
        the power of that value, returning floor of the result. If other
        is an SArray, performs an element-wise power of the two
        arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '**'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '**'))

    def __neg__(self):
        """
        Returns the negative of each element.
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(0, '-'))

    def __pos__(self):
        if self.dtype not in [int, long, float, array.array]:
            raise RuntimeError("Runtime Exception. Unsupported type operation. "
                               "cannot perform operation + on type %s" % str(self.dtype))

        with cython_context():
            return SArray(_proxy = self.__proxy__)

    def __abs__(self):
        """
        Returns the absolute value of each element.
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.left_scalar_operator(0, 'left_abs'))

    def __mod__(self, other):
        """
        Other must be a scalar value. Performs an element wise division remainder.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '%'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '%'))


    def __lt__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '<'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '<'))

    def __gt__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '>'))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '>'))


    def __le__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '<='))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '<='))


    def __ge__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '>='))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '>='))


    def __radd__(self, other):
        """
        Adds a scalar value to the current array.
        Returned array has the same type as the array on the right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '+'))


    def __rsub__(self, other):
        """
        Subtracts a scalar value from the current array.
        Returned array has the same type as the array on the right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '-'))


    def __rmul__(self, other):
        """
        Multiplies a scalar value to the current array.
        Returned array has the same type as the array on the right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '*'))


    def __rdiv__(self, other):
        """
        Divides a scalar value by each element in the array
        Returned array has the same type as the array on the right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '/'))

    def __rtruediv__(self, other):
        """
        Divides a scalar value by each element in the array
        Returned array has the same type as the array on the right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '/'))

    def __rfloordiv__(self, other):
        """
        Divides a scalar value by each element in the array returning the
        floored result.  Returned array has the same type as the array on the
        right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '/')).astype(int)


    def __rmod__(self, other):
        """
        Divides a scalar value by each element in the array returning the
        floored result.  Returned array has the same type as the array on the
        right hand side
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '%'))

    def __rpow__(self, other):
        """
        Raises each element of the current array to the power of that
        value, returning floor of the result.
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.right_scalar_operator(other, '**'))

    def __eq__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the new result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '=='))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '=='))


    def __ne__(self, other):
        """
        If other is a scalar value, compares each element of the current array
        by the value, returning the new result. If other is an SArray, performs
        an element-wise comparison of the two arrays.
        """
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '!='))
            else:
                return SArray(_proxy = self.__proxy__.left_scalar_operator(other, '!='))


    def __and__(self, other):
        """
        Perform a logical element-wise 'and' against another SArray.
        """
        if type(other) is SArray:
            with cython_context():
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '&'))
        else:
            raise TypeError("SArray can only perform logical and against another SArray")


    def __or__(self, other):
        """
        Perform a logical element-wise 'or' against another SArray.
        """
        if type(other) is SArray:
            with cython_context():
                return SArray(_proxy = self.__proxy__.vector_operator(other.__proxy__, '|'))
        else:
            raise TypeError("SArray can only perform logical or against another SArray")


    def __has_size__(self):
        """
        Returns whether or not the size of the SArray is known.
        """
        return self.__proxy__.has_size()

    def __getitem__(self, other):
        """
        If the key is an SArray of identical length, this function performs a
        logical filter: i.e. it subselects all the elements in this array
        where the corresponding value in the other array evaluates to true.
        If the key is an integer this returns a single row of
        the SArray. If the key is a slice, this returns an SArray with the
        sliced rows. See the Turi Create User Guide for usage examples.
        """
        if isinstance(other, numbers.Integral):
            sa_len = len(self)
            if other < 0:
                other += sa_len
            if other >= sa_len:
                raise IndexError("SArray index out of range")

            try:
                lb, ub, value_list = self._getitem_cache
                if lb <= other < ub:
                    return value_list[other - lb]

            except AttributeError:
                pass

            # Not in cache, need to grab it
            block_size = 1024 * (32 if self.dtype in [int, long, float] else 4)
            if self.dtype in [numpy.ndarray, _Image, dict, list]:
                block_size = 16

            block_num = int(other // block_size)

            lb = block_num * block_size
            ub = min(sa_len, lb + block_size)

            val_list = list(SArray(_proxy = self.__proxy__.copy_range(lb, 1, ub)))
            self._getitem_cache = (lb, ub, val_list)
            return val_list[other - lb]

        elif type(other) is SArray:
            if self.__has_size__() and other.__has_size__() and len(other) != len(self):
                raise IndexError("Cannot perform logical indexing on arrays of different length.")
            with cython_context():
                return SArray(_proxy = self.__proxy__.logical_filter(other.__proxy__))

        elif type(other) is slice:
            sa_len = len(self)
            start = other.start
            stop = other.stop
            step = other.step
            if start is None:
                start = 0
            if stop is None:
                stop = sa_len
            if step is None:
                step = 1
            # handle negative indices
            if start < 0:
                start = sa_len + start
            if stop < 0:
                stop = sa_len + stop

            return SArray(_proxy = self.__proxy__.copy_range(start, step, stop))
        else:
            raise IndexError("Invalid type to use for indexing")

    def materialize(self):
        """
        For a SArray that is lazily evaluated, force persist this sarray
        to disk, committing all lazy evaluated operations.
        """
        return self.__materialize__()

    def __materialize__(self):
        """
        For a SArray that is lazily evaluated, force persist this sarray
        to disk, committing all lazy evaluated operations.
        """
        with cython_context():
            self.__proxy__.materialize()

    def is_materialized(self):
        """
        Returns whether or not the sarray has been materialized.
        """
        return self.__is_materialized__()

    def __is_materialized__(self):
        """
        Returns whether or not the sarray has been materialized.
        """
        return self.__proxy__.is_materialized()

    @property
    def dtype(self):
        """
        The data type of the SArray.

        Returns
        -------
        out : type
            The type of the SArray.

        Examples
        --------
        >>> sa = tc.SArray(["The quick brown fox jumps over the lazy dog."])
        >>> sa.dtype
        str
        >>> sa = tc.SArray(range(10))
        >>> sa.dtype
        int
        """
        return self.__proxy__.dtype()

    def head(self, n=10):
        """
        Returns an SArray which contains the first n rows of this SArray.

        Parameters
        ----------
        n : int
            The number of rows to fetch.

        Returns
        -------
        out : SArray
            A new SArray which contains the first n rows of the current SArray.

        Examples
        --------
        >>> tc.SArray(range(10)).head(5)
        dtype: int
        Rows: 5
        [0, 1, 2, 3, 4]
        """
        return SArray(_proxy=self.__proxy__.head(n))

    def vector_slice(self, start, end=None):
        """
        If this SArray contains vectors or lists, this returns a new SArray
        containing each individual element sliced, between start and
        end (exclusive).

        Parameters
        ----------
        start : int
            The start position of the slice.

        end : int, optional.
            The end position of the slice. Note that the end position
            is NOT included in the slice. Thus a g.vector_slice(1,3) will extract
            entries in position 1 and 2. If end is not specified, the return
            array will contain only one element, the element at the start
            position.

        Returns
        -------
        out : SArray
            Each individual vector sliced according to the arguments.

        Examples
        --------

        If g is a vector of floats:

        >>> g = SArray([[1,2,3],[2,3,4]])
        >>> g
        dtype: array
        Rows: 2
        [array('d', [1.0, 2.0, 3.0]), array('d', [2.0, 3.0, 4.0])]

        >>> g.vector_slice(0) # extracts the first element of each vector
        dtype: float
        Rows: 2
        [1.0, 2.0]

        >>> g.vector_slice(0, 2) # extracts the first two elements of each vector
        dtype: array.array
        Rows: 2
        [array('d', [1.0, 2.0]), array('d', [2.0, 3.0])]

        If a vector cannot be sliced, the result will be None:

        >>> g = SArray([[1],[1,2],[1,2,3]])
        >>> g
        dtype: array.array
        Rows: 3
        [array('d', [1.0]), array('d', [1.0, 2.0]), array('d', [1.0, 2.0, 3.0])]

        >>> g.vector_slice(2)
        dtype: float
        Rows: 3
        [None, None, 3.0]

        >>> g.vector_slice(0,2)
        dtype: list
        Rows: 3
        [None, array('d', [1.0, 2.0]), array('d', [1.0, 2.0])]

        If g is a vector of mixed types (float, int, str, array, list, etc.):

        >>> g = SArray([['a',1,1.0],['b',2,2.0]])
        >>> g
        dtype: list
        Rows: 2
        [['a', 1, 1.0], ['b', 2, 2.0]]

        >>> g.vector_slice(0) # extracts the first element of each vector
        dtype: list
        Rows: 2
        [['a'], ['b']]
        """
        if (self.dtype != array.array) and (self.dtype != list):
            raise RuntimeError("Only Vector type can be sliced")
        if end is None:
            end = start + 1

        with cython_context():
            return SArray(_proxy=self.__proxy__.vector_slice(start, end))

    def element_slice(self, start=None, stop=None, step=None):
        """
        This returns an SArray with each element sliced accordingly to the
        slice specified. This is conceptually equivalent to:

        >>> g.apply(lambda x: x[start:step:stop])

        The SArray must be of type list, vector, or string.

        For instance:

        >>> g = SArray(["abcdef","qwerty"])
        >>> g.element_slice(start=0, stop=2)
        dtype: str
        Rows: 2
        ["ab", "qw"]
        >>> g.element_slice(3,-1)
        dtype: str
        Rows: 2
        ["de", "rt"]
        >>> g.element_slice(3)
        dtype: str
        Rows: 2
        ["def", "rty"]

        >>> g = SArray([[1,2,3], [4,5,6]])
        >>> g.element_slice(0, 1)
        dtype: str
        Rows: 2
        [[1], [4]]

        Parameters
        ----------
        start : int or None (default)
            The start position of the slice

        stop: int or None (default)
            The stop position of the slice

        step: int or None (default)
            The step size of the slice

        Returns
        -------
        out : SArray
            Each individual vector/string/list sliced according to the arguments.

        """
        if self.dtype not in [str, array.array, list]:
            raise TypeError("SArray must contain strings, arrays or lists")
        with cython_context():
            return SArray(_proxy=self.__proxy__.subslice(start, step, stop))

    def _count_words(self, to_lower=True, delimiters=["\r", "\v", "\n", "\f", "\t", " "]):
        """
        This returns an SArray with, for each input string, a dict from the unique,
        delimited substrings to their number of occurrences within the original
        string.

        The SArray must be of type string.

        ..WARNING:: This function is deprecated, and will be removed in future
        versions of Turi Create. Please use the `text_analytics.count_words`
        function instead.

        Parameters
        ----------
        to_lower : bool, optional
            "to_lower" indicates whether to map the input strings to lower case
            before counts

        delimiters: list[string], optional
            "delimiters" is a list of which characters to delimit on to find tokens

        Returns
        -------
        out : SArray
            for each input string, a dict from the unique, delimited substrings
            to their number of occurrences within the original string.

        Examples
        --------
        >>> sa = turicreate.SArray(["The quick brown fox jumps.",
                                 "Word word WORD, word!!!word"])
        >>> sa._count_words()
        dtype: dict
        Rows: 2
        [{'quick': 1, 'brown': 1, 'jumps': 1, 'fox': 1, 'the': 1},
         {'word': 2, 'word,': 1, 'word!!!word': 1}]
            """

        if (self.dtype != str):
            raise TypeError("Only SArray of string type is supported for counting bag of words")

        if (not all([len(delim) == 1 for delim in delimiters])):
            raise ValueError("Delimiters must be single-character strings")


        # construct options, will extend over time
        options = dict()
        options["to_lower"] = to_lower == True
        # defaults to std::isspace whitespace delimiters if no others passed in
        options["delimiters"] = delimiters

        with cython_context():
            return SArray(_proxy=self.__proxy__.count_bag_of_words(options))

    def _count_ngrams(self, n=2, method="word", to_lower=True, ignore_space=True):
        """
        For documentation, see turicreate.text_analytics.count_ngrams().

        ..WARNING:: This function is deprecated, and will be removed in future
        versions of Turi Create. Please use the `text_analytics.count_words`
        function instead.
        """
        if (self.dtype != str):
            raise TypeError("Only SArray of string type is supported for counting n-grams")

        if (type(n) != int):
            raise TypeError("Input 'n' must be of type int")

        if (n < 1):
            raise ValueError("Input 'n' must be greater than 0")

        if (n > 5):
            warnings.warn("It is unusual for n-grams to be of size larger than 5.")


        # construct options, will extend over time
        options = dict()
        options["to_lower"] = to_lower == True
        options["ignore_space"] = ignore_space == True


        if method == "word":
            with cython_context():
                return SArray(_proxy=self.__proxy__.count_ngrams(n, options))
        elif method == "character" :
            with cython_context():
                return SArray(_proxy=self.__proxy__.count_character_ngrams(n, options))
        else:
            raise ValueError("Invalid 'method' input  value. Please input either 'word' or 'character' ")

    def dict_trim_by_keys(self, keys, exclude=True):
        """
        Filter an SArray of dictionary type by the given keys. By default, all
        keys that are in the provided list in ``keys`` are *excluded* from the
        returned SArray.

        Parameters
        ----------
        keys : list
            A collection of keys to trim down the elements in the SArray.

        exclude : bool, optional
            If True, all keys that are in the input key list are removed. If
            False, only keys that are in the input key list are retained.

        Returns
        -------
        out : SArray
            A SArray of dictionary type, with each dictionary element trimmed
            according to the input criteria.

        See Also
        --------
        dict_trim_by_values

        Examples
        --------
        >>> sa = turicreate.SArray([{"this":1, "is":1, "dog":2},
                                  {"this": 2, "are": 2, "cat": 1}])
        >>> sa.dict_trim_by_keys(["this", "is", "and", "are"], exclude=True)
        dtype: dict
        Rows: 2
        [{'dog': 2}, {'cat': 1}]
        """
        if not _is_non_string_iterable(keys):
            keys = [keys]


        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_trim_by_keys(keys, exclude))

    def dict_trim_by_values(self, lower=None, upper=None):
        """
        Filter dictionary values to a given range (inclusive). Trimming is only
        performed on values which can be compared to the bound values. Fails on
        SArrays whose data type is not ``dict``.

        Parameters
        ----------
        lower : int or long or float, optional
            The lowest dictionary value that would be retained in the result. If
            not given, lower bound is not applied.

        upper : int or long or float, optional
            The highest dictionary value that would be retained in the result.
            If not given, upper bound is not applied.

        Returns
        -------
        out : SArray
            An SArray of dictionary type, with each dict element trimmed
            according to the input criteria.

        See Also
        --------
        dict_trim_by_keys

        Examples
        --------
        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},
                                  {"this": 2, "are": 1, "cat": 5}])
        >>> sa.dict_trim_by_values(2,5)
        dtype: dict
        Rows: 2
        [{'is': 5}, {'this': 2, 'cat': 5}]

        >>> sa.dict_trim_by_values(upper=5)
        dtype: dict
        Rows: 2
        [{'this': 1, 'is': 5}, {'this': 2, 'are': 1, 'cat': 5}]
        """

        if not (lower is None or isinstance(lower, numbers.Number)):
            raise TypeError("lower bound has to be a numeric value")

        if not (upper is None or isinstance(upper, numbers.Number)):
            raise TypeError("upper bound has to be a numeric value")


        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_trim_by_values(lower, upper))

    def dict_keys(self):
        """
        Create an SArray that contains all the keys from each dictionary
        element as a list. Fails on SArrays whose data type is not ``dict``.

        Returns
        -------
        out : SArray
            A SArray of list type, where each element is a list of keys
            from the input SArray element.

        See Also
        --------
        dict_values

        Examples
        ---------
        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},
                                  {"this": 2, "are": 1, "cat": 5}])
        >>> sa.dict_keys()
        dtype: list
        Rows: 2
        [['this', 'is', 'dog'], ['this', 'are', 'cat']]
        """

        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_keys())

    def dict_values(self):
        """
        Create an SArray that contains all the values from each dictionary
        element as a list. Fails on SArrays whose data type is not ``dict``.

        Returns
        -------
        out : SArray
            A SArray of list type, where each element is a list of values
            from the input SArray element.

        See Also
        --------
        dict_keys

        Examples
        --------
        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},
                                 {"this": 2, "are": 1, "cat": 5}])
        >>> sa.dict_values()
        dtype: list
        Rows: 2
        [[1, 5, 7], [2, 1, 5]]

        """

        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_values())

    def dict_has_any_keys(self, keys):
        """
        Create a boolean SArray by checking the keys of an SArray of
        dictionaries. An element of the output SArray is True if the
        corresponding input element's dictionary has any of the given keys.
        Fails on SArrays whose data type is not ``dict``.

        Parameters
        ----------
        keys : list
            A list of key values to check each dictionary against.

        Returns
        -------
        out : SArray
            A SArray of int type, where each element indicates whether the
            input SArray element contains any key in the input list.

        See Also
        --------
        dict_has_all_keys

        Examples
        --------
        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7}, {"animal":1},
                                 {"this": 2, "are": 1, "cat": 5}])
        >>> sa.dict_has_any_keys(["is", "this", "are"])
        dtype: int
        Rows: 3
        [1, 0, 1]
        """
        if not _is_non_string_iterable(keys):
            keys = [keys]


        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_has_any_keys(keys))

    def dict_has_all_keys(self, keys):
        """
        Create a boolean SArray by checking the keys of an SArray of
        dictionaries. An element of the output SArray is True if the
        corresponding input element's dictionary has all of the given keys.
        Fails on SArrays whose data type is not ``dict``.

        Parameters
        ----------
        keys : list
            A list of key values to check each dictionary against.

        Returns
        -------
        out : SArray
            A SArray of int type, where each element indicates whether the
            input SArray element contains all keys in the input list.

        See Also
        --------
        dict_has_any_keys

        Examples
        --------
        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},
                                 {"this": 2, "are": 1, "cat": 5}])
        >>> sa.dict_has_all_keys(["is", "this"])
        dtype: int
        Rows: 2
        [1, 0]
        """
        if not _is_non_string_iterable(keys):
            keys = [keys]


        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_has_all_keys(keys))

    def apply(self, fn, dtype=None, skip_na=True, seed=None):
        """
        apply(fn, dtype=None, skip_na=True, seed=None)

        Transform each element of the SArray by a given function. The result
        SArray is of type ``dtype``. ``fn`` should be a function that returns
        exactly one value which can be cast into the type specified by
        ``dtype``. If ``dtype`` is not specified, the first 100 elements of the
        SArray are used to make a guess about the data type.

        Parameters
        ----------
        fn : function
            The function to transform each element. Must return exactly one
            value which can be cast into the type specified by ``dtype``.
            This can also be a toolkit extension function which is compiled
            as a native shared library using SDK.


        dtype : {None, int, float, str, list, array.array, dict, turicreate.Image}, optional
            The data type of the new SArray. If ``None``, the first 100 elements
            of the array are used to guess the target data type.

        skip_na : bool, optional
            If True, will not apply ``fn`` to any undefined values.

        seed : int, optional
            Used as the seed if a random number generator is included in ``fn``.

        Returns
        -------
        out : SArray
            The SArray transformed by ``fn``. Each element of the SArray is of
            type ``dtype``.

        See Also
        --------
        SFrame.apply

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.apply(lambda x: x*2)
        dtype: int
        Rows: 3
        [2, 4, 6]

        Using native toolkit extension function:

        .. code-block:: c++

            #include <turicreate/sdk/toolkit_function_macros.hpp>
            #include <cmath>

            using namespace turi;
            double logx(const flexible_type& x, double base) {
              return log((double)(x)) / log(base);
            }

            BEGIN_FUNCTION_REGISTRATION
            REGISTER_FUNCTION(logx, "x", "base");
            END_FUNCTION_REGISTRATION

        compiled into example.so

        >>> import example

        >>> sa = turicreate.SArray([1,2,4])
        >>> sa.apply(lambda x: example.logx(x, 2))
        dtype: float
        Rows: 3
        [0.0, 1.0, 2.0]
        """
        assert callable(fn), "Input function must be callable."

        dryrun = [fn(i) for i in self.head(100) if i is not None]
        if dtype is None:
            dtype = infer_type_of_list(dryrun)
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)

        # log metric

        # First phase test if it is a toolkit function
        nativefn = None
        try:
            from .. import extensions
            nativefn = extensions._build_native_function_call(fn)
        except:
            # failure are fine. we just fall out into the next few phases
            pass

        if nativefn is not None:
            # this is a toolkit lambda. We can do something about it
            nativefn.native_fn_name = nativefn.native_fn_name.encode()
            with cython_context():
                return SArray(_proxy=self.__proxy__.transform_native(nativefn, dtype, skip_na, seed))

        with cython_context():
            return SArray(_proxy=self.__proxy__.transform(fn, dtype, skip_na, seed))


    def filter(self, fn, skip_na=True, seed=None):
        """
        Filter this SArray by a function.

        Returns a new SArray filtered by this SArray.  If `fn` evaluates an
        element to true, this element is copied to the new SArray. If not, it
        isn't. Throws an exception if the return type of `fn` is not castable
        to a boolean value.

        Parameters
        ----------
        fn : function
            Function that filters the SArray. Must evaluate to bool or int.

        skip_na : bool, optional
            If True, will not apply fn to any undefined values.

        seed : int, optional
            Used as the seed if a random number generator is included in fn.

        Returns
        -------
        out : SArray
            The SArray filtered by fn. Each element of the SArray is of
            type int.

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.filter(lambda x: x < 3)
        dtype: int
        Rows: 2
        [1, 2]
        """
        assert callable(fn), "Input must be callable"
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)


        with cython_context():
            return SArray(_proxy=self.__proxy__.filter(fn, skip_na, seed))


    def sample(self, fraction, seed=None, exact=False):
        """
        Create an SArray which contains a subsample of the current SArray.

        Parameters
        ----------
        fraction : float
            Fraction of the rows to fetch. Must be between 0 and 1.
            if exact is False (default), the number of rows returned is
            approximately the fraction times the number of rows.

        seed : int, optional
            The random seed for the random number generator.

        exact: bool, optional
            Defaults to False. If exact=True, an exact fraction is returned, 
            but at a performance penalty.

        Returns
        -------
        out : SArray
            The new SArray which contains the subsampled rows.

        Examples
        --------
        >>> sa = turicreate.SArray(range(10))
        >>> sa.sample(.3)
        dtype: int
        Rows: 3
        [2, 6, 9]
        """
        if (fraction > 1 or fraction < 0):
            raise ValueError('Invalid sampling rate: ' + str(fraction))
        if (len(self) == 0):
            return SArray()
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)


        with cython_context():
            return SArray(_proxy=self.__proxy__.sample(fraction, seed, exact))

    def hash(self, seed=0):
        """
        Returns an SArray with a hash of each element. seed can be used
        to change the hash function to allow this method to be used for
        random number generation.

        Parameters
        ----------
        seed : int
            Defaults to 0. Can be changed to different values to get
            different hash results.

        Returns
        -------
        out : SArray
            An integer SArray with a hash value for each element. Identical
            elements are hashed to the same value
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.hash(seed))

    @classmethod
    def random_integers(cls, size, seed=None):
        """
        Returns an SArray with random integer values.
        """
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)
        return cls.from_sequence(size).hash(seed)

    def _save_as_text(self, url):
        """
        Save the SArray to disk as text file.
        """
        raise NotImplementedError


    def all(self):
        """
        Return True if every element of the SArray evaluates to True. For
        numeric SArrays zeros and missing values (``None``) evaluate to False,
        while all non-zero, non-missing values evaluate to True. For string,
        list, and dictionary SArrays, empty values (zero length strings, lists
        or dictionaries) or missing values (``None``) evaluate to False. All
        other values evaluate to True.

        Returns True on an empty SArray.

        Returns
        -------
        out : bool

        See Also
        --------
        any

        Examples
        --------
        >>> turicreate.SArray([1, None]).all()
        False
        >>> turicreate.SArray([1, 0]).all()
        False
        >>> turicreate.SArray([1, 2]).all()
        True
        >>> turicreate.SArray(["hello", "world"]).all()
        True
        >>> turicreate.SArray(["hello", ""]).all()
        False
        >>> turicreate.SArray([]).all()
        True
        """
        with cython_context():
            return self.__proxy__.all()


    def any(self):
        """
        Return True if any element of the SArray evaluates to True. For numeric
        SArrays any non-zero value evaluates to True. For string, list, and
        dictionary SArrays, any element of non-zero length evaluates to True.

        Returns False on an empty SArray.

        Returns
        -------
        out : bool

        See Also
        --------
        all

        Examples
        --------
        >>> turicreate.SArray([1, None]).any()
        True
        >>> turicreate.SArray([1, 0]).any()
        True
        >>> turicreate.SArray([0, 0]).any()
        False
        >>> turicreate.SArray(["hello", "world"]).any()
        True
        >>> turicreate.SArray(["hello", ""]).any()
        True
        >>> turicreate.SArray(["", ""]).any()
        False
        >>> turicreate.SArray([]).any()
        False
        """
        with cython_context():
            return self.__proxy__.any()


    def max(self):
        """
        Get maximum numeric value in SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type.

        Returns
        -------
        out : type of SArray
            Maximum value of SArray

        See Also
        --------
        min

        Examples
        --------
        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).max()
        96
        """
        with cython_context():
            return self.__proxy__.max()


    def min(self):
        """
        Get minimum numeric value in SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type.

        Returns
        -------
        out : type of SArray
            Minimum value of SArray

        See Also
        --------
        max

        Examples
        --------
        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).min()

        """
        with cython_context():
            return self.__proxy__.min()

    def argmax(self):
        """
        Get the index of the maximum numeric value in SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type.

        Returns
        -------
        out : int
            Index of the maximum value of SArray

        See Also
        --------
        argmin

        Examples
        --------
        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).argmax()

        """
        from .sframe import SFrame as _SFrame

        if len(self) == 0:
            return None
        if not any([isinstance(self[0], i) for i in [int,float,long]]):
            raise TypeError("SArray must be of type 'int', 'long', or 'float'.")

        sf = _SFrame(self).add_row_number()
        sf_out = sf.groupby(key_column_names=[],operations={'maximum_x1': _aggregate.ARGMAX('X1','id')})
        return sf_out['maximum_x1'][0]

    def argmin(self):
        """
        Get the index of the minimum numeric value in SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type.

        Returns
        -------
        out : int
            index of the minimum value of SArray

        See Also
        --------
        argmax

        Examples
        --------
        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).argmin()

        """
        from .sframe import SFrame as _SFrame

        if len(self) == 0:
            return None
        if not any([isinstance(self[0], i) for i in [int,float,long]]):
            raise TypeError("SArray must be of type 'int', 'long', or 'float'.")

        sf = _SFrame(self).add_row_number()
        sf_out = sf.groupby(key_column_names=[],operations={'minimum_x1': _aggregate.ARGMIN('X1','id')})
        return sf_out['minimum_x1'][0]


    def sum(self):
        """
        Sum of all values in this SArray.

        Raises an exception if called on an SArray of strings, lists, or
        dictionaries. If the SArray contains numeric arrays (array.array) and
        all the arrays are the same length, the sum over all the arrays will be
        returned. Returns None on an empty SArray. For large values, this may
        overflow without warning.

        Returns
        -------
        out : type of SArray
            Sum of all values in SArray
        """
        with cython_context():
            return self.__proxy__.sum()

    def mean(self):
        """
        Mean of all the values in the SArray, or mean image.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type or non-Image type.

        Returns
        -------
        out : float | turicreate.Image
            Mean of all values in SArray, or image holding per-pixel mean
            across the input SArray.
        """
        with cython_context():
            if self.dtype == _Image:
                from  .. import extensions
                return extensions.generate_mean(self)
            else:
                return self.__proxy__.mean()


    def std(self, ddof=0):
        """
        Standard deviation of all the values in the SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type or if `ddof` >= length of SArray.

        Parameters
        ----------
        ddof : int, optional
            "delta degrees of freedom" in the variance calculation.

        Returns
        -------
        out : float
            The standard deviation of all the values.
        """
        with cython_context():
            return self.__proxy__.std(ddof)


    def var(self, ddof=0):
        """
        Variance of all the values in the SArray.

        Returns None on an empty SArray. Raises an exception if called on an
        SArray with non-numeric type or if `ddof` >= length of SArray.

        Parameters
        ----------
        ddof : int, optional
            "delta degrees of freedom" in the variance calculation.

        Returns
        -------
        out : float
            Variance of all values in SArray.
        """
        with cython_context():
            return self.__proxy__.var(ddof)

    def countna(self):
        """
        Number of missing elements in the SArray.

        Returns
        -------
        out : int
            Number of missing values.
        """
        with cython_context():
            return self.__proxy__.num_missing()

    def nnz(self):
        """
        Number of non-zero elements in the SArray.

        Returns
        -------
        out : int
            Number of non-zero elements.
        """
        with cython_context():
            return self.__proxy__.nnz()

    def datetime_to_str(self,format="%Y-%m-%dT%H:%M:%S%ZP"):
        """
        Create a new SArray with all the values cast to str. The string format is
        specified by the 'format' parameter.

        Parameters
        ----------
        format : str
            The format to output the string. Default format is "%Y-%m-%dT%H:%M:%S%ZP".

        Returns
        -------
        out : SArray[str]
            The SArray converted to the type 'str'.

        Examples
        --------
        >>> dt = datetime.datetime(2011, 10, 20, 9, 30, 10, tzinfo=GMT(-5))
        >>> sa = turicreate.SArray([dt])
        >>> sa.datetime_to_str("%e %b %Y %T %ZP")
        dtype: str
        Rows: 1
        [20 Oct 2011 09:30:10 GMT-05:00]

        See Also
        ----------
        str_to_datetime

        References
        ----------
        [1] Boost date time from string conversion guide (http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html)

        """
        if(self.dtype != datetime.datetime):
            raise TypeError("datetime_to_str expects SArray of datetime as input SArray")

        with cython_context():
            return SArray(_proxy=self.__proxy__.datetime_to_str(format))

    def str_to_datetime(self,format="%Y-%m-%dT%H:%M:%S%ZP"):
        """
        Create a new SArray with all the values cast to datetime. The string format is
        specified by the 'format' parameter.

        Parameters
        ----------
        format : str
            The string format of the input SArray. Default format is "%Y-%m-%dT%H:%M:%S%ZP".
            If format is "ISO", the the format is "%Y%m%dT%H%M%S%F%q"
        Returns
        -------
        out : SArray[datetime.datetime]
            The SArray converted to the type 'datetime'.

        Examples
        --------
        >>> sa = turicreate.SArray(["20-Oct-2011 09:30:10 GMT-05:30"])
        >>> sa.str_to_datetime("%d-%b-%Y %H:%M:%S %ZP")
        dtype: datetime
        Rows: 1
        datetime.datetime(2011, 10, 20, 9, 30, 10, tzinfo=GMT(-5.5))

        See Also
        ----------
        datetime_to_str

        References
        ----------
        [1] boost date time to string conversion guide (http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html)

        """
        if(self.dtype != str):
            raise TypeError("str_to_datetime expects SArray of str as input SArray")

        with cython_context():
            return SArray(_proxy=self.__proxy__.str_to_datetime(format))

    def pixel_array_to_image(self, width, height, channels, undefined_on_failure=True, allow_rounding=False):
        """
        Create a new SArray with all the values cast to :py:class:`turicreate.image.Image`
        of uniform size.

        Parameters
        ----------
        width: int
            The width of the new images.

        height: int
            The height of the new images.

        channels: int.
            Number of channels of the new images.

        undefined_on_failure: bool , optional , default True
            If True, return None type instead of Image type in failure instances.
            If False, raises error upon failure.

        allow_rounding: bool, optional , default False
            If True, rounds non-integer values when converting to Image type.
            If False, raises error upon rounding.

        Returns
        -------
        out : SArray[turicreate.Image]
            The SArray converted to the type 'turicreate.Image'.

        See Also
        --------
        astype, str_to_datetime, datetime_to_str

        Examples
        --------
        The MNIST data is scaled from 0 to 1, but our image type only loads integer  pixel values
        from 0 to 255. If we just convert without scaling, all values below one would be cast to
        0.

        >>> mnist_array = turicreate.SArray('https://static.turi.com/datasets/mnist/mnist_vec_sarray')
        >>> scaled_mnist_array = mnist_array * 255
        >>> mnist_img_sarray = tc.SArray.pixel_array_to_image(scaled_mnist_array, 28, 28, 1, allow_rounding = True)

        """
        if(self.dtype != array.array):
            raise TypeError("array_to_img expects SArray of arrays as input SArray")

        num_to_test = 10

        num_test = min(len(self), num_to_test)

        mod_values = [val % 1 for x in range(num_test) for val in self[x]]

        out_of_range_values = [(val > 255 or val < 0) for x in range(num_test) for val in self[x]]

        if sum(mod_values) != 0.0 and not allow_rounding:
            raise ValueError("There are non-integer values in the array data. Images only support integer data values between 0 and 255. To permit rounding, set the 'allow_rounding' parameter to 1.")

        if sum(out_of_range_values) != 0:
            raise ValueError("There are values outside the range of 0 to 255. Images only support integer data values between 0 and 255.")


        from .. import extensions
        return extensions.vector_sarray_to_image_sarray(self, width, height, channels, undefined_on_failure)

    def astype(self, dtype, undefined_on_failure=False):
        """
        Create a new SArray with all values cast to the given type. Throws an
        exception if the types are not castable to the given type.

        Parameters
        ----------
        dtype : {int, float, str, list, array.array, dict, datetime.datetime}
            The type to cast the elements to in SArray

        undefined_on_failure: bool, optional
            If set to True, runtime cast failures will be emitted as missing
            values rather than failing.

        Returns
        -------
        out : SArray [dtype]
            The SArray converted to the type ``dtype``.

        Notes
        -----
        - The string parsing techniques used to handle conversion to dictionary
          and list types are quite generic and permit a variety of interesting
          formats to be interpreted. For instance, a JSON string can usually be
          interpreted as a list or a dictionary type. See the examples below.
        - For datetime-to-string  and string-to-datetime conversions,
          use sa.datetime_to_str() and sa.str_to_datetime() functions.
        - For array.array to turicreate.Image conversions, use sa.pixel_array_to_image()

        Examples
        --------
        >>> sa = turicreate.SArray(['1','2','3','4'])
        >>> sa.astype(int)
        dtype: int
        Rows: 4
        [1, 2, 3, 4]

        Given an SArray of strings that look like dicts, convert to a dictionary
        type:

        >>> sa = turicreate.SArray(['{1:2 3:4}', '{a:b c:d}'])
        >>> sa.astype(dict)
        dtype: dict
        Rows: 2
        [{1: 2, 3: 4}, {'a': 'b', 'c': 'd'}]
        """

        if (dtype == _Image) and (self.dtype == array.array):
            raise TypeError("Cannot cast from image type to array with sarray.astype(). Please use sarray.pixel_array_to_img() instead.")

        with cython_context():
            return SArray(_proxy=self.__proxy__.astype(dtype, undefined_on_failure))

    def clip(self, lower=float('nan'), upper=float('nan')):
        """
        Create a new SArray with each value clipped to be within the given
        bounds.

        In this case, "clipped" means that values below the lower bound will be
        set to the lower bound value. Values above the upper bound will be set
        to the upper bound value. This function can operate on SArrays of
        numeric type as well as array type, in which case each individual
        element in each array is clipped. By default ``lower`` and ``upper`` are
        set to ``float('nan')`` which indicates the respective bound should be
        ignored. The method fails if invoked on an SArray of non-numeric type.

        Parameters
        ----------
        lower : int, optional
            The lower bound used to clip. Ignored if equal to ``float('nan')``
            (the default).

        upper : int, optional
            The upper bound used to clip. Ignored if equal to ``float('nan')``
            (the default).

        Returns
        -------
        out : SArray

        See Also
        --------
        clip_lower, clip_upper

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.clip(2,2)
        dtype: int
        Rows: 3
        [2, 2, 2]
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(lower, upper))

    def clip_lower(self, threshold):
        """
        Create new SArray with all values clipped to the given lower bound. This
        function can operate on numeric arrays, as well as vector arrays, in
        which case each individual element in each vector is clipped. Throws an
        exception if the SArray is empty or the types are non-numeric.

        Parameters
        ----------
        threshold : float
            The lower bound used to clip values.

        Returns
        -------
        out : SArray

        See Also
        --------
        clip, clip_upper

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.clip_lower(2)
        dtype: int
        Rows: 3
        [2, 2, 3]
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(threshold, float('nan')))


    def clip_upper(self, threshold):
        """
        Create new SArray with all values clipped to the given upper bound. This
        function can operate on numeric arrays, as well as vector arrays, in
        which case each individual element in each vector is clipped.

        Parameters
        ----------
        threshold : float
            The upper bound used to clip values.

        Returns
        -------
        out : SArray

        See Also
        --------
        clip, clip_lower

        Examples
        --------
        >>> sa = turicreate.SArray([1,2,3])
        >>> sa.clip_upper(2)
        dtype: int
        Rows: 3
        [1, 2, 2]
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(float('nan'), threshold))

    def tail(self, n=10):
        """
        Get an SArray that contains the last n elements in the SArray.

        Parameters
        ----------
        n : int
            The number of elements to fetch

        Returns
        -------
        out : SArray
            A new SArray which contains the last n rows of the current SArray.
        """
        with cython_context():
            return SArray(_proxy=self.__proxy__.tail(n))


    def dropna(self):
        """
        Create new SArray containing only the non-missing values of the
        SArray.

        A missing value shows up in an SArray as 'None'.  This will also drop
        float('nan').

        Returns
        -------
        out : SArray
            The new SArray with missing values removed.
        """


        with cython_context():
            return SArray(_proxy = self.__proxy__.drop_missing_values())

    def fillna(self, value):
        """
        Create new SArray with all missing values (None or NaN) filled in
        with the given value.

        The size of the new SArray will be the same as the original SArray. If
        the given value is not the same type as the values in the SArray,
        `fillna` will attempt to convert the value to the original SArray's
        type. If this fails, an error will be raised.

        Parameters
        ----------
        value : type convertible to SArray's type
            The value used to replace all missing values

        Returns
        -------
        out : SArray
            A new SArray with all missing values filled
        """

        with cython_context():
            return SArray(_proxy = self.__proxy__.fill_missing_values(value))

    def is_topk(self, topk=10, reverse=False):
        """
        Create an SArray indicating which elements are in the top k.

        Entries are '1' if the corresponding element in the current SArray is a
        part of the top k elements, and '0' if that corresponding element is
        not. Order is descending by default.

        Parameters
        ----------
        topk : int
            The number of elements to determine if 'top'

        reverse : bool
            If True, return the topk elements in ascending order

        Returns
        -------
        out : SArray (of type int)

        Notes
        -----
        This is used internally by SFrame's topk function.
        """
        with cython_context():
            return SArray(_proxy = self.__proxy__.topk_index(topk, reverse))

    def summary(self, background=False, sub_sketch_keys=None):
        """
        Summary statistics that can be calculated with one pass over the SArray.

        Returns a turicreate.Sketch object which can be further queried for many
        descriptive statistics over this SArray. Many of the statistics are
        approximate. See the :class:`~turicreate.Sketch` documentation for more
        detail.

        Parameters
        ----------
        background : boolean, optional
          If True, the sketch construction will return immediately and the
          sketch will be constructed in the background. While this is going on,
          the sketch can be queried incrementally, but at a performance penalty.
          Defaults to False.

        sub_sketch_keys : int | str | list of int | list of str, optional
            For SArray of dict type, also constructs sketches for a given set of keys,
            For SArray of array type, also constructs sketches for the given indexes.
            The sub sketches may be queried using: :py:func:`~turicreate.Sketch.element_sub_sketch()`.
            Defaults to None in which case no subsketches will be constructed.

        Returns
        -------
        out : Sketch
            Sketch object that contains descriptive statistics for this SArray.
            Many of the statistics are approximate.
        """
        from ..data_structures.sketch import Sketch
        if (self.dtype == _Image):
            raise TypeError("summary() is not supported for arrays of image type")
        if (type(background) != bool):
            raise TypeError("'background' parameter has to be a boolean value")
        if (sub_sketch_keys is not None):
            if (self.dtype != dict and self.dtype != array.array):
                raise TypeError("sub_sketch_keys is only supported for SArray of dictionary or array type")
            if not _is_non_string_iterable(sub_sketch_keys):
                sub_sketch_keys = [sub_sketch_keys]
            value_types = set([type(i) for i in sub_sketch_keys])
            if (len(value_types) != 1):
                raise ValueError("sub_sketch_keys member values need to have the same type.")
            value_type = value_types.pop()
            if (self.dtype == dict and value_type != str):
                raise TypeError("Only string value(s) can be passed to sub_sketch_keys for SArray of dictionary type. "+
                    "For dictionary types, sketch summary is computed by casting keys to string values.")
            if (self.dtype == array.array and value_type != int):
                raise TypeError("Only int value(s) can be passed to sub_sketch_keys for SArray of array type")
        else:
            sub_sketch_keys = list()

        return Sketch(self, background, sub_sketch_keys = sub_sketch_keys)

    def value_counts(self):
        """
        Return an SFrame containing counts of unique values. The resulting
        SFrame will be sorted in descending frequency.

        Returns
        -------
        out : SFrame
            An SFrame containing 2 columns : 'value', and 'count'. The SFrame will
            be sorted in descending order by the column 'count'.

        See Also
        --------
        SFrame.summary

        Examples
        --------
        >>> sa = turicreate.SArray([1,1,2,2,2,2,3,3,3,3,3,3,3])
        >>> sa.value_counts()
            Columns:
                    value	int
                    count	int
            Rows: 3
            Data:
            +-------+-------+
            | value | count |
            +-------+-------+
            |   3   |   7   |
            |   2   |   4   |
            |   1   |   2   |
            +-------+-------+
            [3 rows x 2 columns]
        """
        from .sframe import SFrame as _SFrame
        return _SFrame({'value':self}).groupby('value', {'count':_aggregate.COUNT}).sort('count', ascending=False)


    def append(self, other):
        """
        Append an SArray to the current SArray. Creates a new SArray with the
        rows from both SArrays. Both SArrays must be of the same type.

        Parameters
        ----------
        other : SArray
            Another SArray whose rows are appended to current SArray.

        Returns
        -------
        out : SArray
            A new SArray that contains rows from both SArrays, with rows from
            the ``other`` SArray coming after all rows from the current SArray.

        See Also
        --------
        SFrame.append

        Examples
        --------
        >>> sa = turicreate.SArray([1, 2, 3])
        >>> sa2 = turicreate.SArray([4, 5, 6])
        >>> sa.append(sa2)
        dtype: int
        Rows: 6
        [1, 2, 3, 4, 5, 6]
        """
        if type(other) is not SArray:
            raise RuntimeError("SArray append can only work with SArray")

        if self.dtype != other.dtype:
            raise RuntimeError("Data types in both SArrays have to be the same")

        with cython_context():
            return SArray(_proxy = self.__proxy__.append(other.__proxy__))

    def unique(self):
        """
        Get all unique values in the current SArray.

        Raises a TypeError if the SArray is of dictionary type. Will not
        necessarily preserve the order of the given SArray in the new SArray.


        Returns
        -------
        out : SArray
            A new SArray that contains the unique values of the current SArray.

        See Also
        --------
        SFrame.unique
        """
        from .sframe import SFrame as _SFrame

        tmp_sf = _SFrame()
        tmp_sf.add_column(self, 'X1', inplace=True)

        res = tmp_sf.groupby('X1',{})

        return SArray(_proxy=res['X1'].__proxy__)

    def explore(self, title=None):
        """
        Explore the SArray in an interactive GUI. Opens a new app window.

        Parameters
        ----------
        title : str
            The plot title to show for the resulting visualization. Defaults to None.
            If the title is None, a default title will be provided.

        Returns
        -------
        None

        Examples
        --------
        Suppose 'sa' is an SArray, we can view it using:

        >>> sa.explore()

        To override the default plot title and axis labels:

        >>> sa.explore(title="My Plot Title")
        """
        from .sframe import SFrame as _SFrame
        _SFrame({'SArray': self}).explore()

    def show(self, title=None, xlabel=None, ylabel=None):
        """
        Visualize the SArray.

        Notes
        -----
        - The plot will render either inline in a Jupyter Notebook, or in a
          native GUI window, depending on the value provided in
          `turicreate.visualization.set_target` (defaults to 'auto').

        Parameters
        ----------
        title : str
            The plot title to show for the resulting visualization. Defaults to None.
            If the title is None, a default title will be provided.

        xlabel : str
            The X axis label to show for the resulting visualization. Defaults to None.
            If the xlabel is None, a default X axis label will be provided.

        ylabel : str
            The Y axis label to show for the resulting visualization. Defaults to None.
            If the ylabel is None, a default Y axis label will be provided.

        Returns
        -------
        None

        Examples
        --------
        Suppose 'sa' is an SArray, we can view it using:

        >>> sa.show()

        To override the default plot title and axis labels:

        >>> sa.show(title="My Plot Title", xlabel="My X Axis", ylabel="My Y Axis")
        """

        returned_plot = self.plot(title, xlabel, ylabel)

        returned_plot.show()

    def plot(self, title=None, xlabel=None, ylabel=None):
        """
        Create a Plot object representing the SArray.

        Notes
        -----
        - The plot will render either inline in a Jupyter Notebook, or in a
          native GUI window, depending on the value provided in
          `turicreate.visualization.set_target` (defaults to 'auto').

        Parameters
        ----------
        title : str
            The plot title to show for the resulting visualization. Defaults to None.
            If the title is None, a default title will be provided.

        xlabel : str
            The X axis label to show for the resulting visualization. Defaults to None.
            If the xlabel is None, a default X axis label will be provided.

        ylabel : str
            The Y axis label to show for the resulting visualization. Defaults to None.
            If the ylabel is None, a default Y axis label will be provided.

        Returns
        -------
        out : Plot
        A :class: Plot object that is the visualization of the SArray.

        Examples
        --------
        Suppose 'sa' is an SArray, we can create a plot of it using:

        >>> plt = sa.plot()

        To override the default plot title and axis labels:

        >>> plt = sa.plot(title="My Plot Title", xlabel="My X Axis", ylabel="My Y Axis")

        We can then visualize the plot using:

        >>> plt.show()
        """
        path_to_client = _get_client_app_path()

        if title == "":
            title = " "
        if xlabel == "":
            xlabel = " "
        if ylabel == "":
            ylabel = " "

        if title is None:
            title = "" # C++ otherwise gets "None" as std::string
        if xlabel is None:
            xlabel = ""
        if ylabel is None:
            ylabel = ""

        return Plot(self.__proxy__.plot(path_to_client, title, xlabel, ylabel))

    def item_length(self):
        """
        Length of each element in the current SArray.

        Only works on SArrays of dict, array, or list type. If a given element
        is a missing value, then the output elements is also a missing value.
        This function is equivalent to the following but more performant:

            sa_item_len =  sa.apply(lambda x: len(x) if x is not None else None)

        Returns
        -------
        out_sf : SArray
            A new SArray, each element in the SArray is the len of the corresponding
            items in original SArray.

        Examples
        --------
        >>> sa = SArray([
        ...  {"is_restaurant": 1, "is_electronics": 0},
        ...  {"is_restaurant": 1, "is_retail": 1, "is_electronics": 0},
        ...  {"is_restaurant": 0, "is_retail": 1, "is_electronics": 0},
        ...  {"is_restaurant": 0},
        ...  {"is_restaurant": 1, "is_electronics": 1},
        ...  None])
        >>> sa.item_length()
        dtype: int
        Rows: 6
        [2, 3, 3, 1, 2, None]
        """
        if (self.dtype not in [list, dict, array.array]):
            raise TypeError("item_length() is only applicable for SArray of type list, dict and array.")


        with cython_context():
            return SArray(_proxy = self.__proxy__.item_length())

    def random_split(self, fraction, seed=None):
        """
        Randomly split the rows of an SArray into two SArrays. The first SArray
        contains *M* rows, sampled uniformly (without replacement) from the
        original SArray. *M* is approximately the fraction times the original
        number of rows. The second SArray contains the remaining rows of the
        original SArray.

        Parameters
        ----------
        fraction : float
            Approximate fraction of the rows to fetch for the first returned
            SArray. Must be between 0 and 1.

        seed : int, optional
            Seed for the random number generator used to split.

        Returns
        -------
        out : tuple [SArray]
            Two new SArrays.

        Examples
        --------
        Suppose we have an SArray with 1,024 rows and we want to randomly split
        it into training and testing datasets with about a 90%/10% split.

        >>> sa = turicreate.SArray(range(1024))
        >>> sa_train, sa_test = sa.random_split(.9, seed=5)
        >>> print(len(sa_train), len(sa_test))
        922 102
        """
        from .sframe import SFrame
        temporary_sf = SFrame()
        temporary_sf['X1'] = self
        (train, test) = temporary_sf.random_split(fraction, seed)
        return (train['X1'], test['X1'])

    def split_datetime(self, column_name_prefix = "X", limit=None, timezone=False):
        """
        Splits an SArray of datetime type to multiple columns, return a
        new SFrame that contains expanded columns. A SArray of datetime will be
        split by default into an SFrame of 6 columns, one for each
        year/month/day/hour/minute/second element.

        **Column Naming**

        When splitting a SArray of datetime type, new columns are named:
        prefix.year, prefix.month, etc. The prefix is set by the parameter
        "column_name_prefix" and defaults to 'X'. If column_name_prefix is
        None or empty, then no prefix is used.

        **Timezone Column**
        If timezone parameter is True, then timezone information is represented
        as one additional column which is a float shows the offset from
        GMT(0.0) or from UTC.


        Parameters
        ----------
        column_name_prefix: str, optional
            If provided, expanded column names would start with the given prefix.
            Defaults to "X".

        limit: list[str], optional
            Limits the set of datetime elements to expand.
            Possible values are 'year','month','day','hour','minute','second',
            'weekday', 'isoweekday', 'tmweekday', and 'us'.
            If not provided, only ['year','month','day','hour','minute','second']
            are expanded.

            - 'year': The year number
            - 'month': A value between 1 and 12 where 1 is January.
            - 'day': Day of the months. Begins at 1.
            - 'hour': Hours since midnight.
            - 'minute': Minutes after the hour.
            - 'second': Seconds after the minute.
            - 'us': Microseconds after the second. Between 0 and 999,999.
            - 'weekday': A value between 0 and 6 where 0 is Monday.
            - 'isoweekday': A value between 1 and 7 where 1 is Monday.
            - 'tmweekday': A value between 0 and 7 where 0 is Sunday

        timezone: bool, optional
            A boolean parameter that determines whether to show timezone column or not.
            Defaults to False.

        Returns
        -------
        out : SFrame
            A new SFrame that contains all expanded columns

        Examples
        --------
        To expand only day and year elements of a datetime SArray

         >>> sa = SArray(
            [datetime(2011, 1, 21, 7, 7, 21, tzinfo=GMT(0)),
             datetime(2010, 2, 5, 7, 8, 21, tzinfo=GMT(4.5)])

         >>> sa.split_datetime(column_name_prefix=None,limit=['day','year'])
            Columns:
                day   int
                year  int
            Rows: 2
            Data:
            +-------+--------+
            |  day  |  year  |
            +-------+--------+
            |   21  |  2011  |
            |   5   |  2010  |
            +-------+--------+
            [2 rows x 2 columns]


        To expand only year and timezone elements of a datetime SArray
        with timezone column represented as a string. Columns are named with prefix:
        'Y.column_name'.

        >>> sa.split_datetime(column_name_prefix="Y",limit=['year'],timezone=True)
            Columns:
                Y.year  int
                Y.timezone float
            Rows: 2
            Data:
            +----------+---------+
            |  Y.year  | Y.timezone |
            +----------+---------+
            |    2011  |  0.0    |
            |    2010  |  4.5    |
            +----------+---------+
            [2 rows x 2 columns]
        """
        from .sframe import SFrame as _SFrame

        if self.dtype != datetime.datetime:
            raise TypeError("Only column of datetime type is supported.")

        if column_name_prefix is None:
            column_name_prefix = ""
        if six.PY2 and type(column_name_prefix) == unicode:
            column_name_prefix = column_name_prefix.encode('utf-8')
        if type(column_name_prefix) != str:
            raise TypeError("'column_name_prefix' must be a string")

        # convert limit to column_keys
        if limit is not None:
            if not _is_non_string_iterable(limit):
                raise TypeError("'limit' must be a list")

            name_types = set([type(i) for i in limit])
            if (len(name_types) != 1):
                raise TypeError("'limit' contains values that are different types")

            if (name_types.pop() != str):
                raise TypeError("'limit' must contain string values.")

            if len(set(limit)) != len(limit):
                raise ValueError("'limit' contains duplicate values")

        column_types = []

        if(limit is None):
            limit = ['year','month','day','hour','minute','second']

        column_types = [int] * len(limit)

        if(timezone == True):
            limit += ['timezone']
            column_types += [float]


        with cython_context():
           return _SFrame(_proxy=self.__proxy__.expand(column_name_prefix, limit, column_types))

    def stack(self, new_column_name=None, drop_na=False, new_column_type=None):
        """
        Convert a "wide" SArray to one or two "tall" columns in an SFrame by
        stacking all values.

        The stack works only for columns of dict, list, or array type.  If the
        column is dict type, two new columns are created as a result of
        stacking: one column holds the key and another column holds the value.
        The rest of the columns are repeated for each key/value pair.

        If the column is array or list type, one new column is created as a
        result of stacking. With each row holds one element of the array or list
        value, and the rest columns from the same original row repeated.

        The returned SFrame includes the newly created column(s).

        Parameters
        --------------
        new_column_name : str | list of str, optional
            The new column name(s). If original column is list/array type,
            new_column_name must a string. If original column is dict type,
            new_column_name must be a list of two strings. If not given, column
            names are generated automatically.

        drop_na : boolean, optional
            If True, missing values and empty list/array/dict are all dropped
            from the resulting column(s). If False, missing values are
            maintained in stacked column(s).

        new_column_type : type | list of types, optional
            The new column types. If original column is a list/array type
            new_column_type must be a single type, or a list of one type. If
            original column is of dict type, new_column_type must be a list of
            two types. If not provided, the types are automatically inferred
            from the first 100 values of the SFrame.

        Returns
        -------
        out : SFrame
            A new SFrame that contains the newly stacked column(s).

        Examples
        ---------
        Suppose 'sa' is an SArray of dict type:

        >>> sa = turicreate.SArray([{'a':3, 'cat':2},
        ...                         {'a':1, 'the':2},
        ...                         {'the':1, 'dog':3},
        ...                         {}])
        [{'a': 3, 'cat': 2}, {'a': 1, 'the': 2}, {'the': 1, 'dog': 3}, {}]

        Stack would stack all keys in one column and all values in another
        column:

        >>> sa.stack(new_column_name=['word', 'count'])
        +------+-------+
        | word | count |
        +------+-------+
        |  a   |   3   |
        | cat  |   2   |
        |  a   |   1   |
        | the  |   2   |
        | the  |   1   |
        | dog  |   3   |
        | None |  None |
        +------+-------+
        [7 rows x 2 columns]

        Observe that since topic 4 had no words, an empty row is inserted.
        To drop that row, set drop_na=True in the parameters to stack.
        """
        from .sframe import SFrame as _SFrame
        return _SFrame({'SArray': self}).stack('SArray',
                                               new_column_name=new_column_name,
                                               drop_na=drop_na,
                                               new_column_type=new_column_type)

    def unpack(self, column_name_prefix = "X", column_types=None, na_value=None, limit=None):
        """
        Convert an SArray of list, array, or dict type to an SFrame with
        multiple columns.

        `unpack` expands an SArray using the values of each list/array/dict as
        elements in a new SFrame of multiple columns. For example, an SArray of
        lists each of length 4 will be expanded into an SFrame of 4 columns,
        one for each list element. An SArray of lists/arrays of varying size
        will be expand to a number of columns equal to the longest list/array.
        An SArray of dictionaries will be expanded into as many columns as
        there are keys.

        When unpacking an SArray of list or array type, new columns are named:
        `column_name_prefix`.0, `column_name_prefix`.1, etc. If unpacking a
        column of dict type, unpacked columns are named
        `column_name_prefix`.key1, `column_name_prefix`.key2, etc.

        When unpacking an SArray of list or dictionary types, missing values in
        the original element remain as missing values in the resultant columns.
        If the `na_value` parameter is specified, all values equal to this
        given value are also replaced with missing values. In an SArray of
        array.array type, NaN is interpreted as a missing value.

        :py:func:`turicreate.SFrame.pack_columns()` is the reverse effect of unpack

        Parameters
        ----------
        column_name_prefix: str, optional
            If provided, unpacked column names would start with the given prefix.

        column_types: list[type], optional
            Column types for the unpacked columns. If not provided, column
            types are automatically inferred from first 100 rows. Defaults to
            None.

        na_value: optional
            Convert all values that are equal to `na_value` to
            missing value if specified.

        limit: list, optional
            Limits the set of list/array/dict keys to unpack.
            For list/array SArrays, 'limit' must contain integer indices.
            For dict SArray, 'limit' must contain dictionary keys.

        Returns
        -------
        out : SFrame
            A new SFrame that contains all unpacked columns

        Examples
        --------
        To unpack a dict SArray

        >>> sa = SArray([{ 'word': 'a',     'count': 1},
        ...              { 'word': 'cat',   'count': 2},
        ...              { 'word': 'is',    'count': 3},
        ...              { 'word': 'coming','count': 4}])

        Normal case of unpacking SArray of type dict:

        >>> sa.unpack(column_name_prefix=None)
        Columns:
            count   int
            word    str
        <BLANKLINE>
        Rows: 4
        <BLANKLINE>
        Data:
        +-------+--------+
        | count |  word  |
        +-------+--------+
        |   1   |   a    |
        |   2   |  cat   |
        |   3   |   is   |
        |   4   | coming |
        +-------+--------+
        [4 rows x 2 columns]
        <BLANKLINE>

        Unpack only keys with 'word':

        >>> sa.unpack(limit=['word'])
        Columns:
            X.word  str
        <BLANKLINE>
        Rows: 4
        <BLANKLINE>
        Data:
        +--------+
        | X.word |
        +--------+
        |   a    |
        |  cat   |
        |   is   |
        | coming |
        +--------+
        [4 rows x 1 columns]
        <BLANKLINE>

        >>> sa2 = SArray([
        ...               [1, 0, 1],
        ...               [1, 1, 1],
        ...               [0, 1]])

        Convert all zeros to missing values:

        >>> sa2.unpack(column_types=[int, int, int], na_value=0)
        Columns:
            X.0     int
            X.1     int
            X.2     int
        <BLANKLINE>
        Rows: 3
        <BLANKLINE>
        Data:
        +------+------+------+
        | X.0  | X.1  | X.2  |
        +------+------+------+
        |  1   | None |  1   |
        |  1   |  1   |  1   |
        | None |  1   | None |
        +------+------+------+
        [3 rows x 3 columns]
        <BLANKLINE>
        """
        from .sframe import SFrame as _SFrame

        if self.dtype not in [dict, array.array, list]:
            raise TypeError("Only SArray of dict/list/array type supports unpack")

        if column_name_prefix is None:
            column_name_prefix = ""
        if not(isinstance(column_name_prefix, six.string_types)):
            raise TypeError("'column_name_prefix' must be a string")

        # validate 'limit'
        if limit is not None:
            if (not _is_non_string_iterable(limit)):
                raise TypeError("'limit' must be a list")

            name_types = set([type(i) for i in limit])
            if (len(name_types) != 1):
                raise TypeError("'limit' contains values that are different types")

            # limit value should be numeric if unpacking sarray.array value
            if (self.dtype != dict) and (name_types.pop() != int):
                raise TypeError("'limit' must contain integer values.")

            if len(set(limit)) != len(limit):
                raise ValueError("'limit' contains duplicate values")

        if (column_types is not None):
            if not _is_non_string_iterable(column_types):
                raise TypeError("column_types must be a list")

            for column_type in column_types:
                if (column_type not in (int, float, str, list, dict, array.array)):
                    raise TypeError("column_types contains unsupported types. Supported types are ['float', 'int', 'list', 'dict', 'str', 'array.array']")

            if limit is not None:
                if len(limit) != len(column_types):
                    raise ValueError("limit and column_types do not have the same length")
            elif self.dtype == dict:
                raise ValueError("if 'column_types' is given, 'limit' has to be provided to unpack dict type.")
            else:
                limit = range(len(column_types))

        else:
            head_rows = self.head(100).dropna()
            lengths = [len(i) for i in head_rows]
            if len(lengths) == 0 or max(lengths) == 0:
                raise RuntimeError("Cannot infer number of items from the SArray, SArray may be empty. please explicitly provide column types")

            # infer column types for dict type at server side, for list and array, infer from client side
            if self.dtype != dict:
                length = max(lengths)
                if limit is None:
                    limit = range(length)
                else:
                    # adjust the length
                    length = len(limit)

                if self.dtype == array.array:
                    column_types = [float for i in range(length)]
                else:
                    column_types = list()
                    for i in limit:
                        t = [(x[i] if ((x is not None) and len(x) > i) else None) for x in head_rows]
                        column_types.append(infer_type_of_list(t))


        with cython_context():
            if (self.dtype == dict and column_types is None):
                limit = limit if limit is not None else []
                return _SFrame(_proxy=self.__proxy__.unpack_dict(column_name_prefix.encode('utf-8'), limit, na_value))
            else:
                return _SFrame(_proxy=self.__proxy__.unpack(column_name_prefix.encode('utf-8'), limit, column_types, na_value))

    def sort(self, ascending=True):
        """
        Sort all values in this SArray.

        Sort only works for sarray of type str, int and float, otherwise TypeError
        will be raised. Creates a new, sorted SArray.

        Parameters
        ----------
        ascending: boolean, optional
           If true, the sarray values are sorted in ascending order, otherwise,
           descending order.

        Returns
        -------
        out: SArray

        Examples
        --------
        >>> sa = SArray([3,2,1])
        >>> sa.sort()
        dtype: int
        Rows: 3
        [1, 2, 3]
        """
        from .sframe import SFrame as _SFrame

        if self.dtype not in (int, float, str, datetime.datetime):
            raise TypeError("Only sarray with type (int, float, str, datetime.datetime) can be sorted")
        sf = _SFrame()
        sf['a'] = self
        return sf.sort('a', ascending)['a']

    def __check_min_observations(self, min_observations):
        if min_observations is None:
            min_observations = (1 << 64) - 1
        if min_observations < 0:
            raise ValueError("min_observations must be a positive integer")
        return min_observations

    def rolling_mean(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the mean of different subsets over this
        SArray.

        Also known as a "moving average" or "running average". The subset that
        the mean is calculated over is defined as an inclusive range relative
        to the position to each value in the SArray, using `window_start` and
        `window_end`. For a better understanding of this, see the examples
        below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the mean relative to the
            current value.

        window_end : int
            The end of the subset to calculate the mean relative to the current
            value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the mean (otherwise result is None). None signifies that
            the entire window must not include a missing value. A negative
            number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling mean with a window including the previous 2 entries including
        the current:
        >>> sa.rolling_mean(-2,0)
        dtype: float
        Rows: 5
        [None, None, 2.0, 3.0, 4.0]

        Pandas equivalent:
        >>> pandas.rolling_mean(series, 3)
        0   NaN
        1   NaN
        2     2
        3     3
        4     4
        dtype: float64

        Same rolling mean operation, but 2 minimum observations:
        >>> sa.rolling_mean(-2,0,min_observations=2)
        dtype: float
        Rows: 5
        [None, 1.5, 2.0, 3.0, 4.0]

        Pandas equivalent:
        >>> pandas.rolling_mean(series, 3, min_periods=2)
        0    NaN
        1    1.5
        2    2.0
        3    3.0
        4    4.0
        dtype: float64

        A rolling mean with a size of 3, centered around the current:
        >>> sa.rolling_mean(-1,1)
        dtype: float
        Rows: 5
        [None, 2.0, 3.0, 4.0, None]

        Pandas equivalent:
        >>> pandas.rolling_mean(series, 3, center=True)
        0   NaN
        1     2
        2     3
        3     4
        4   NaN
        dtype: float64

        A rolling mean with a window including the current and the 2 entries
        following:
        >>> sa.rolling_mean(0,2)
        dtype: float
        Rows: 5
        [2.0, 3.0, 4.0, None, None]

        A rolling mean with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_mean(-2,-1)
        dtype: float
        Rows: 5
        [None, None, 1.5, 2.5, 3.5]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = None
        if self.dtype is array.array:
            agg_op = '__builtin__vector__avg__'
        else:
            agg_op = '__builtin__avg__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_sum(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the sum of different subsets over this
        SArray.

        Also known as a "moving sum" or "running sum". The subset that
        the sum is calculated over is defined as an inclusive range relative
        to the position to each value in the SArray, using `window_start` and
        `window_end`. For a better understanding of this, see the examples
        below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the sum relative to the
            current value.

        window_end : int
            The end of the subset to calculate the sum relative to the current
            value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the sum (otherwise result is None). None signifies that
            the entire window must not include a missing value. A negative
            number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling sum with a window including the previous 2 entries including
        the current:
        >>> sa.rolling_sum(-2,0)
        dtype: int
        Rows: 5
        [None, None, 6, 9, 12]

        Pandas equivalent:
        >>> pandas.rolling_sum(series, 3)
        0   NaN
        1   NaN
        2     6
        3     9
        4    12
        dtype: float64

        Same rolling sum operation, but 2 minimum observations:
        >>> sa.rolling_sum(-2,0,min_observations=2)
        dtype: int
        Rows: 5
        [None, 3, 6, 9, 12]

        Pandas equivalent:
        >>> pandas.rolling_sum(series, 3, min_periods=2)
        0    NaN
        1      3
        2      6
        3      9
        4     12
        dtype: float64

        A rolling sum with a size of 3, centered around the current:
        >>> sa.rolling_sum(-1,1)
        dtype: int
        Rows: 5
        [None, 6, 9, 12, None]

        Pandas equivalent:
        >>> pandas.rolling_sum(series, 3, center=True)
        0   NaN
        1     6
        2     9
        3    12
        4   NaN
        dtype: float64

        A rolling sum with a window including the current and the 2 entries
        following:
        >>> sa.rolling_sum(0,2)
        dtype: int
        Rows: 5
        [6, 9, 12, None, None]

        A rolling sum with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_sum(-2,-1)
        dtype: int
        Rows: 5
        [None, None, 3, 5, 7]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = None
        if self.dtype is array.array:
            agg_op = '__builtin__vector__sum__'
        else:
            agg_op = '__builtin__sum__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_max(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the maximum value of different subsets over
        this SArray.

        The subset that the maximum is calculated over is defined as an
        inclusive range relative to the position to each value in the SArray,
        using `window_start` and `window_end`. For a better understanding of
        this, see the examples below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the maximum relative to the
            current value.

        window_end : int
            The end of the subset to calculate the maximum relative to the current
            value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the maximum (otherwise result is None). None signifies that
            the entire window must not include a missing value. A negative
            number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling max with a window including the previous 2 entries including
        the current:
        >>> sa.rolling_max(-2,0)
        dtype: int
        Rows: 5
        [None, None, 3, 4, 5]

        Pandas equivalent:
        >>> pandas.rolling_max(series, 3)
        0   NaN
        1   NaN
        2     3
        3     4
        4     5
        dtype: float64

        Same rolling max operation, but 2 minimum observations:
        >>> sa.rolling_max(-2,0,min_observations=2)
        dtype: int
        Rows: 5
        [None, 2, 3, 4, 5]

        Pandas equivalent:
        >>> pandas.rolling_max(series, 3, min_periods=2)
        0    NaN
        1      2
        2      3
        3      4
        4      5
        dtype: float64

        A rolling max with a size of 3, centered around the current:
        >>> sa.rolling_max(-1,1)
        dtype: int
        Rows: 5
        [None, 3, 4, 5, None]

        Pandas equivalent:
        >>> pandas.rolling_max(series, 3, center=True)
        0   NaN
        1     3
        2     4
        3     5
        4   NaN
        dtype: float64

        A rolling max with a window including the current and the 2 entries
        following:
        >>> sa.rolling_max(0,2)
        dtype: int
        Rows: 5
        [3, 4, 5, None, None]

        A rolling max with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_max(-2,-1)
        dtype: int
        Rows: 5
        [None, None, 2, 3, 4]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__max__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_min(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the minimum value of different subsets over
        this SArray.

        The subset that the minimum is calculated over is defined as an
        inclusive range relative to the position to each value in the SArray,
        using `window_start` and `window_end`. For a better understanding of
        this, see the examples below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the minimum relative to the
            current value.

        window_end : int
            The end of the subset to calculate the minimum relative to the current
            value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the minimum (otherwise result is None). None signifies that
            the entire window must not include a missing value. A negative
            number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling min with a window including the previous 2 entries including
        the current:
        >>> sa.rolling_min(-2,0)
        dtype: int
        Rows: 5
        [None, None, 1, 2, 3]

        Pandas equivalent:
        >>> pandas.rolling_min(series, 3)
        0   NaN
        1   NaN
        2     1
        3     2
        4     3
        dtype: float64

        Same rolling min operation, but 2 minimum observations:
        >>> sa.rolling_min(-2,0,min_observations=2)
        dtype: int
        Rows: 5
        [None, 1, 1, 2, 3]

        Pandas equivalent:
        >>> pandas.rolling_min(series, 3, min_periods=2)
        0    NaN
        1      1
        2      1
        3      2
        4      3
        dtype: float64

        A rolling min with a size of 3, centered around the current:
        >>> sa.rolling_min(-1,1)
        dtype: int
        Rows: 5
        [None, 1, 2, 3, None]

        Pandas equivalent:
        >>> pandas.rolling_min(series, 3, center=True)
        0   NaN
        1     1
        2     2
        3     3
        4   NaN
        dtype: float64

        A rolling min with a window including the current and the 2 entries
        following:
        >>> sa.rolling_min(0,2)
        dtype: int
        Rows: 5
        [1, 2, 3, None, None]

        A rolling min with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_min(-2,-1)
        dtype: int
        Rows: 5
        [None, None, 1, 2, 3]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__min__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_var(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the variance of different subsets over this
        SArray.

        The subset that the variance is calculated over is defined as an inclusive
        range relative to the position to each value in the SArray, using
        `window_start` and `window_end`. For a better understanding of this,
        see the examples below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the variance relative to the
            current value.

        window_end : int
            The end of the subset to calculate the variance relative to the current
            value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the variance (otherwise result is None). None signifies that
            the entire window must not include a missing value. A negative
            number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling variance with a window including the previous 2 entries
        including the current:
        >>> sa.rolling_var(-2,0)
        dtype: float
        Rows: 5
        [None, None, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666]

        Pandas equivalent:
        >>> pandas.rolling_var(series, 3, ddof=0)
        0         NaN
        1         NaN
        2    0.666667
        3    0.666667
        4    0.666667
        dtype: float64

        Same rolling variance operation, but 2 minimum observations:
        >>> sa.rolling_var(-2,0,min_observations=2)
        dtype: float
        Rows: 5
        [None, 0.25, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666]

        Pandas equivalent:
        >>> pandas.rolling_var(series, 3, ddof=0, min_periods=2)
        0         NaN
        1    0.250000
        2    0.666667
        3    0.666667
        4    0.666667
        dtype: float64

        A rolling variance with a size of 3, centered around the current:
        >>> sa.rolling_var(-1,1)
        dtype: float
        Rows: 5
        [None, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, None]

        Pandas equivalent:
        >>> pandas.rolling_var(series, 3, center=True)
        0         NaN
        1    0.666667
        2    0.666667
        3    0.666667
        4         NaN
        dtype: float64

        A rolling variance with a window including the current and the 2 entries
        following:
        >>> sa.rolling_var(0,2)
        dtype: float
        Rows: 5
        [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, None, None]

        A rolling variance with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_var(-2,-1)
        dtype: float
        Rows: 5
        [None, None, 0.25, 0.25, 0.25]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__var__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_stdv(self, window_start, window_end, min_observations=None):
        """
        Calculate a new SArray of the standard deviation of different subsets
        over this SArray.

        The subset that the standard deviation is calculated over is defined as
        an inclusive range relative to the position to each value in the
        SArray, using `window_start` and `window_end`. For a better
        understanding of this, see the examples below.

        Parameters
        ----------
        window_start : int
            The start of the subset to calculate the standard deviation
            relative to the current value.

        window_end : int
            The end of the subset to calculate the standard deviation relative
            to the current value. Must be greater than `window_start`.

        min_observations : int
            Minimum number of non-missing observations in window required to
            calculate the standard deviation (otherwise result is None). None
            signifies that the entire window must not include a missing value.
            A negative number throws an error.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,4,5])
        >>> series = pandas.Series([1,2,3,4,5])

        A rolling standard deviation with a window including the previous 2
        entries including the current:
        >>> sa.rolling_stdv(-2,0)
        dtype: float
        Rows: 5
        [None, None, 0.816496580927726, 0.816496580927726, 0.816496580927726]

        Pandas equivalent:
        >>> pandas.rolling_std(series, 3, ddof=0)
        0         NaN
        1         NaN
        2    0.816497
        3    0.816497
        4    0.816497
        dtype: float64

        Same rolling standard deviation operation, but 2 minimum observations:
        >>> sa.rolling_stdv(-2,0,min_observations=2)
        dtype: float
        Rows: 5
        [None, 0.5, 0.816496580927726, 0.816496580927726, 0.816496580927726]

        Pandas equivalent:
        >>> pandas.rolling_std(series, 3, ddof=0, min_periods=2)
        0         NaN
        1    0.500000
        2    0.816497
        3    0.816497
        4    0.816497
        dtype: float64

        A rolling standard deviation with a size of 3, centered around the
        current:
        >>> sa.rolling_stdv(-1,1)
        dtype: float
        Rows: 5
        [None, 0.816496580927726, 0.816496580927726, 0.816496580927726, None]

        Pandas equivalent:
        >>> pandas.rolling_std(series, 3, center=True, ddof=0)
        0         NaN
        1    0.816497
        2    0.816497
        3    0.816497
        4         NaN
        dtype: float64

        A rolling standard deviation with a window including the current and
        the 2 entries following:
        >>> sa.rolling_stdv(0,2)
        dtype: float
        Rows: 5
        [0.816496580927726, 0.816496580927726, 0.816496580927726, None, None]

        A rolling standard deviation with a window including the previous 2
        entries NOT including the current:
        >>> sa.rolling_stdv(-2,-1)
        dtype: float
        Rows: 5
        [None, None, 0.5, 0.5, 0.5]
        """
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__stdv__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_count(self, window_start, window_end):
        """
        Count the number of non-NULL values of different subsets over this
        SArray.

        The subset that the count is executed on is defined as an inclusive
        range relative to the position to each value in the SArray, using
        `window_start` and `window_end`. For a better understanding of this,
        see the examples below.

        Parameters
        ----------
        window_start : int
            The start of the subset to count relative to the current value.

        window_end : int
            The end of the subset to count relative to the current value. Must
            be greater than `window_start`.

        Returns
        -------
        out : SArray

        Examples
        --------
        >>> import pandas
        >>> sa = SArray([1,2,3,None,5])
        >>> series = pandas.Series([1,2,3,None,5])

        A rolling count with a window including the previous 2 entries including
        the current:
        >>> sa.rolling_count(-2,0)
        dtype: int
        Rows: 5
        [1, 2, 3, 2, 2]

        Pandas equivalent:
        >>> pandas.rolling_count(series, 3)
        0     1
        1     2
        2     3
        3     2
        4     2
        dtype: float64

        A rolling count with a size of 3, centered around the current:
        >>> sa.rolling_count(-1,1)
        dtype: int
        Rows: 5
        [2, 3, 2, 2, 1]

        Pandas equivalent:
        >>> pandas.rolling_count(series, 3, center=True)
        0    2
        1    3
        2    2
        3    2
        4    1
        dtype: float64

        A rolling count with a window including the current and the 2 entries
        following:
        >>> sa.rolling_count(0,2)
        dtype: int
        Rows: 5
        [3, 2, 2, 1, 1]

        A rolling count with a window including the previous 2 entries NOT
        including the current:
        >>> sa.rolling_count(-2,-1)
        dtype: int
        Rows: 5
        [0, 1, 2, 2, 1]
        """
        agg_op = '__builtin__nonnull__count__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, 0))

    def cumulative_sum(self):
        """
        Return the cumulative sum of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        sum of all the elements preceding and including it. The SArray is
        expected to be of numeric type (int, float), or a numeric vector type.

        Returns
        -------
        out : sarray[int, float, array.array]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.
         - For SArray's of type array.array, all entries are expected to
           be of the same size.

        Examples
        --------
        >>> sa = SArray([1, 2, 3, 4, 5])
        >>> sa.cumulative_sum()
        dtype: int
        rows: 3
        [1, 3, 6, 10, 15]
        """
        from .. import extensions
        agg_op = "__builtin__cum_sum__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_mean(self):
        """
        Return the cumulative mean of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        mean value of all the elements preceding and including it. The SArray
        is expected to be of numeric type (int, float), or a numeric vector
        type.

        Returns
        -------
        out : Sarray[float, array.array]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.
         - For SArray's of type array.array, all entries are expected to
           be of the same size.

        Examples
        --------
        >>> sa = SArray([1, 2, 3, 4, 5])
        >>> sa.cumulative_mean()
        dtype: float
        rows: 3
        [1, 1.5, 2, 2.5, 3]
        """
        from .. import extensions
        agg_op = "__builtin__cum_avg__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_min(self):
        """
        Return the cumulative minimum value of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        minimum value of all the elements preceding and including it. The
        SArray is expected to be of numeric type (int, float).

        Returns
        -------
        out : SArray[int, float]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.

        Examples
        --------
        >>> sa = SArray([1, 2, 3, 4, 0])
        >>> sa.cumulative_min()
        dtype: int
        rows: 3
        [1, 1, 1, 1, 0]
        """
        from .. import extensions
        agg_op = "__builtin__cum_min__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_max(self):
        """
        Return the cumulative maximum value of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        maximum value of all the elements preceding and including it. The
        SArray is expected to be of numeric type (int, float).

        Returns
        -------
        out : SArray[int, float]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.

        Examples
        --------
        >>> sa = SArray([1, 0, 3, 4, 2])
        >>> sa.cumulative_max()
        dtype: int
        rows: 3
        [1, 1, 3, 4, 4]
        """
        from .. import extensions
        agg_op = "__builtin__cum_max__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_std(self):
        """
        Return the cumulative standard deviation of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        standard deviation of all the elements preceding and including it. The
        SArray is expected to be of numeric type, or a numeric vector type.

        Returns
        -------
        out : SArray[int, float]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.

        Examples
        --------
        >>> sa = SArray([1, 2, 3, 4, 0])
        >>> sa.cumulative_std()
        dtype: float
        rows: 3
        [0.0, 0.5, 0.816496580927726, 1.118033988749895, 1.4142135623730951]
        """
        from .. import extensions
        agg_op = "__builtin__cum_std__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_var(self):
        """
        Return the cumulative variance of the elements in the SArray.

        Returns an SArray where each element in the output corresponds to the
        variance of all the elements preceding and including it. The SArray is
        expected to be of numeric type, or a numeric vector type.

        Returns
        -------
        out : SArray[int, float]

        Notes
        -----
         - Missing values are ignored while performing the cumulative
           aggregate operation.

        Examples
        --------
        >>> sa = SArray([1, 2, 3, 4, 0])
        >>> sa.cumulative_var()
        dtype: float
        rows: 3
        [0.0, 0.25, 0.6666666666666666, 1.25, 2.0]
        """
        from .. import extensions
        agg_op = "__builtin__cum_var__"
        return SArray(_proxy = self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def __copy__(self):
        """
        Returns a shallow copy of the sarray.
        """
        return SArray(_proxy = self.__proxy__)

    def __deepcopy__(self, memo):
        """
        Returns a deep copy of the sarray. As the data in an SArray is
        immutable, this is identical to __copy__.
        """
        return SArray(_proxy = self.__proxy__)
