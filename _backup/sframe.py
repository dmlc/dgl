# -*- coding: utf-8 -*-
# Copyright © 2017 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
'''
This module defines the SFrame class which provides the
ability to create, access and manipulate a remote scalable dataframe object.

SFrame acts similarly to pandas.DataFrame, but the data is completely immutable
and is stored column wise on the Turi Server side.
'''
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..connect import main as glconnect
from ..cython.cy_flexible_type import infer_type_of_list
from ..cython.context import debug_trace as cython_context
from ..cython.cy_sframe import UnitySFrameProxy
from ..util import _is_non_string_iterable, _make_internal_url
from ..util import infer_dbapi2_types
from ..util import get_module_from_object, pytype_to_printf
from ..visualization import _get_client_app_path
from .sarray import SArray, _create_sequential_sarray
from .. import aggregate
from .image import Image as _Image
from ..deps import pandas, numpy, HAS_PANDAS, HAS_NUMPY
from .grouped_sframe import GroupedSFrame
from ..visualization import Plot

import array
from prettytable import PrettyTable
from textwrap import wrap
import datetime
import time
import itertools
import logging as _logging
import numbers
import sys
import six
import csv

__all__ = ['SFrame']
__LOGGER__ = _logging.getLogger(__name__)

FOOTER_STRS = ['Note: Only the head of the SFrame is printed.',
               'You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.']

LAZY_FOOTER_STRS = ['Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.',
                    'You can use sf.materialize() to force materialization.']

if sys.version_info.major > 2:
    long = int

def load_sframe(filename):
    """
    Load an SFrame. The filename extension is used to determine the format
    automatically. This function is particularly useful for SFrames previously
    saved in binary format. For CSV imports the ``SFrame.read_csv`` function
    provides greater control. If the SFrame is in binary format, ``filename`` is
    actually a directory, created when the SFrame is saved.

    Parameters
    ----------
    filename : string
        Location of the file to load. Can be a local path or a remote URL.

    Returns
    -------
    out : SFrame

    See Also
    --------
    SFrame.save, SFrame.read_csv

    Examples
    --------
    >>> sf = turicreate.SFrame({'id':[1,2,3], 'val':['A','B','C']})
    >>> sf.save('my_sframe')        # 'my_sframe' is a directory
    >>> sf_loaded = turicreate.load_sframe('my_sframe')
    """
    sf = SFrame(data=filename)
    return sf

def _get_global_dbapi_info(dbapi_module, conn):
    """
    Fetches all needed information from the top-level DBAPI module,
    guessing at the module if it wasn't passed as a parameter. Returns a
    dictionary of all the needed variables. This is put in one place to
    make sure the error message is clear if the module "guess" is wrong.
    """
    module_given_msg = "The DBAPI2 module given ({0}) is missing the global\n"+\
    "variable '{1}'. Please make sure you are supplying a module that\n"+\
    "conforms to the DBAPI 2.0 standard (PEP 0249)."
    module_not_given_msg = "Hello! I gave my best effort to find the\n"+\
    "top-level module that the connection object you gave me came from.\n"+\
    "I found '{0}' which doesn't have the global variable '{1}'.\n"+\
    "To avoid this confusion, you can pass the module as a parameter using\n"+\
    "the 'dbapi_module' argument to either from_sql or to_sql."

    if dbapi_module is None:
        dbapi_module = get_module_from_object(conn)
        module_given = False
    else:
        module_given = True

    module_name = dbapi_module.__name__ if hasattr(dbapi_module, '__name__') else None

    needed_vars = ['apilevel','paramstyle','Error','DATETIME','NUMBER','ROWID']
    ret_dict = {}
    ret_dict['module_name'] = module_name

    for i in needed_vars:
        tmp = None
        try:
            tmp = eval("dbapi_module."+i)
        except AttributeError as e:
            # Some DBs don't actually care about types, so they won't define
            # the types. These are the ACTUALLY needed variables though
            if i not in ['apilevel','paramstyle','Error']:
                pass
            elif module_given:
                raise AttributeError(module_given_msg.format(module_name, i))
            else:
                raise AttributeError(module_not_given_msg.format(module_name, i))
        ret_dict[i] = tmp

    try:
        if ret_dict['apilevel'][0:3] != "2.0":
            raise NotImplementedError("Unsupported API version " +\
              str(ret_dict['apilevel']) + ". Only DBAPI 2.0 is supported.")
    except TypeError as e:
        e.message = "Module's 'apilevel' value is invalid."
        raise e

    acceptable_paramstyles = ['qmark','numeric','named','format','pyformat']
    try:
        if ret_dict['paramstyle'] not in acceptable_paramstyles:
            raise TypeError("Module's 'paramstyle' value is invalid.")
    except TypeError as e:
        raise TypeError("Module's 'paramstyle' value is invalid.")

    return ret_dict

def _convert_rows_to_builtin_seq(data):
    # Flexible type expects a builtin type (like list or tuple) for conversion.
    # Some DBAPI modules abstract rows as classes that act as single sequences
    # and this allows these to work with flexible type. list is chosen to allow
    # mutation in case we need to force cast any entries
    if len(data) > 0 and type(data[0]) != list:
        data = [list(row) for row in data]
    return data

# Expects list of tuples
def _force_cast_sql_types(data, result_types, force_cast_cols):
    if len(force_cast_cols) == 0:
        return data

    ret_data = []
    for row in data:
        for idx in force_cast_cols:
            if row[idx] is not None and result_types[idx] != datetime.datetime:
                row[idx] = result_types[idx](row[idx])

        ret_data.append(row)

    return ret_data


class SFrame(object):
    """
    A tabular, column-mutable dataframe object that can scale to big data. The
    data in SFrame is stored column-wise on the Turi Server side, and is
    stored on persistent storage (e.g. disk) to avoid being constrained by
    memory size.  Each column in an SFrame is a size-immutable
    :class:`~turicreate.SArray`, but SFrames are mutable in that columns can be
    added and subtracted with ease.  An SFrame essentially acts as an ordered
    dict of SArrays.

    Currently, we support constructing an SFrame from the following data
    formats:

    * csv file (comma separated value)
    * sframe directory archive (A directory where an sframe was saved
      previously)
    * general text file (with csv parsing options, See :py:meth:`read_csv()`)
    * a Python dictionary
    * pandas.DataFrame
    * JSON

    and from the following sources:

    * your local file system
    * the Turi Server's file system
    * HDFS
    * Amazon S3
    * HTTP(S).

    Only basic examples of construction are covered here. For more information
    and examples, please see the `User Guide <https://apple.github.io/turicreate/docs/user
    guide/index.html#Working_with_data_Tabular_data>`_.

    Parameters
    ----------
    data : array | pandas.DataFrame | string | dict, optional
        The actual interpretation of this field is dependent on the ``format``
        parameter. If ``data`` is an array or Pandas DataFrame, the contents are
        stored in the SFrame. If ``data`` is a string, it is interpreted as a
        file. Files can be read from local file system or urls (local://,
        hdfs://, s3://, http://).

    format : string, optional
        Format of the data. The default, "auto" will automatically infer the
        input data format. The inference rules are simple: If the data is an
        array or a dataframe, it is associated with 'array' and 'dataframe'
        respectively. If the data is a string, it is interpreted as a file, and
        the file extension is used to infer the file format. The explicit
        options are:

        - "auto"
        - "array"
        - "dict"
        - "sarray"
        - "dataframe"
        - "csv"
        - "tsv"
        - "sframe".

    See Also
    --------
    read_csv:
        Create a new SFrame from a csv file. Preferred for text and CSV formats,
        because it has a lot more options for controlling the parser.

    save : Save an SFrame for later use.

    Notes
    -----
    - When reading from HDFS on Linux we must guess the location of your java
      installation. By default, we will use the location pointed to by the
      JAVA_HOME environment variable.  If this is not set, we check many common
      installation paths. You may use two environment variables to override
      this behavior.  TURI_JAVA_HOME allows you to specify a specific java
      installation and overrides JAVA_HOME.  TURI_LIBJVM_DIRECTORY
      overrides all and expects the exact directory that your preferred
      libjvm.so file is located.  Use this ONLY if you'd like to use a
      non-standard JVM.

    Examples
    --------

    >>> import turicreate
    >>> from turicreate import SFrame

    **Construction**

    Construct an SFrame from a dataframe and transfers the dataframe object
    across the network.

    >>> df = pandas.DataFrame()
    >>> sf = SFrame(data=df)

    Construct an SFrame from a local csv file (only works for local server).

    >>> sf = SFrame(data='~/mydata/foo.csv')

    Construct an SFrame from a csv file on Amazon S3. This requires the
    environment variables: *AWS_ACCESS_KEY_ID* and *AWS_SECRET_ACCESS_KEY* to be
    set before the python session started.

    >>> sf = SFrame(data='s3://mybucket/foo.csv')

    Read from HDFS using a specific java installation (environment variable
    only applies when using Linux)

    >>> import os
    >>> os.environ['TURI_JAVA_HOME'] = '/my/path/to/java'
    >>> from turicreate import SFrame
    >>> sf = SFrame("hdfs://mycluster.example.com:8020/user/myname/coolfile.txt")

    An SFrame can be constructed from a dictionary of values or SArrays:

    >>> sf = tc.SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    Or equivalently:

    >>> ids = SArray([1,2,3])
    >>> vals = SArray(['A','B','C'])
    >>> sf = SFrame({'id':ids,'val':vals})

    It can also be constructed from an array of SArrays in which case column
    names are automatically assigned.

    >>> ids = SArray([1,2,3])
    >>> vals = SArray(['A','B','C'])
    >>> sf = SFrame([ids, vals])
    >>> sf
    Columns:
        X1 int
        X2 str
    Rows: 3
    Data:
       X1  X2
    0  1   A
    1  2   B
    2  3   C

    If the SFrame is constructed from a list of values, an SFrame of a single
    column is constructed.

    >>> sf = SFrame([1,2,3])
    >>> sf
    Columns:
        X1 int
    Rows: 3
    Data:
       X1
    0  1
    1  2
    2  3

    **Parsing**

    The :py:func:`turicreate.SFrame.read_csv()` is quite powerful and, can be
    used to import a variety of row-based formats.

    First, some simple cases:

    >>> !cat ratings.csv
    user_id,movie_id,rating
    10210,1,1
    10213,2,5
    10217,2,2
    10102,1,3
    10109,3,4
    10117,5,2
    10122,2,4
    10114,1,5
    10125,1,1
    >>> tc.SFrame.read_csv('ratings.csv')
    Columns:
      user_id   int
      movie_id  int
      rating    int
    Rows: 9
    Data:
    +---------+----------+--------+
    | user_id | movie_id | rating |
    +---------+----------+--------+
    |  10210  |    1     |   1    |
    |  10213  |    2     |   5    |
    |  10217  |    2     |   2    |
    |  10102  |    1     |   3    |
    |  10109  |    3     |   4    |
    |  10117  |    5     |   2    |
    |  10122  |    2     |   4    |
    |  10114  |    1     |   5    |
    |  10125  |    1     |   1    |
    +---------+----------+--------+
    [9 rows x 3 columns]


    Delimiters can be specified, if "," is not the delimiter, for instance
    space ' ' in this case. Only single character delimiters are supported.

    >>> !cat ratings.csv
    user_id movie_id rating
    10210 1 1
    10213 2 5
    10217 2 2
    10102 1 3
    10109 3 4
    10117 5 2
    10122 2 4
    10114 1 5
    10125 1 1
    >>> tc.SFrame.read_csv('ratings.csv', delimiter=' ')

    By default, "NA" or a missing element are interpreted as missing values.

    >>> !cat ratings2.csv
    user,movie,rating
    "tom",,1
    harry,5,
    jack,2,2
    bill,,
    >>> tc.SFrame.read_csv('ratings2.csv')
    Columns:
      user  str
      movie int
      rating    int
    Rows: 4
    Data:
    +---------+-------+--------+
    |   user  | movie | rating |
    +---------+-------+--------+
    |   tom   |  None |   1    |
    |  harry  |   5   |  None  |
    |   jack  |   2   |   2    |
    | missing |  None |  None  |
    +---------+-------+--------+
    [4 rows x 3 columns]

    Furthermore due to the dictionary types and list types, can handle parsing
    of JSON-like formats.

    >>> !cat ratings3.csv
    business, categories, ratings
    "Restaurant 1", [1 4 9 10], {"funny":5, "cool":2}
    "Restaurant 2", [], {"happy":2, "sad":2}
    "Restaurant 3", [2, 11, 12], {}
    >>> tc.SFrame.read_csv('ratings3.csv')
    Columns:
    business    str
    categories  array
    ratings dict
    Rows: 3
    Data:
    +--------------+--------------------------------+-------------------------+
    |   business   |           categories           |         ratings         |
    +--------------+--------------------------------+-------------------------+
    | Restaurant 1 | array('d', [1.0, 4.0, 9.0, ... | {'funny': 5, 'cool': 2} |
    | Restaurant 2 |           array('d')           |  {'sad': 2, 'happy': 2} |
    | Restaurant 3 | array('d', [2.0, 11.0, 12.0])  |            {}           |
    +--------------+--------------------------------+-------------------------+
    [3 rows x 3 columns]

    The list and dictionary parsers are quite flexible and can absorb a
    variety of purely formatted inputs. Also, note that the list and dictionary
    types are recursive, allowing for arbitrary values to be contained.

    All these are valid lists:

    >>> !cat interesting_lists.csv
    list
    []
    [1,2,3]
    [1;2,3]
    [1 2 3]
    [{a:b}]
    ["c",d, e]
    [[a]]
    >>> tc.SFrame.read_csv('interesting_lists.csv')
    Columns:
      list  list
    Rows: 7
    Data:
    +-----------------+
    |       list      |
    +-----------------+
    |        []       |
    |    [1, 2, 3]    |
    |    [1, 2, 3]    |
    |    [1, 2, 3]    |
    |   [{'a': 'b'}]  |
    | ['c', 'd', 'e'] |
    |     [['a']]     |
    +-----------------+
    [7 rows x 1 columns]

    All these are valid dicts:

    >>> !cat interesting_dicts.csv
    dict
    {"classic":1,"dict":1}
    {space:1 separated:1}
    {emptyvalue:}
    {}
    {:}
    {recursive1:[{a:b}]}
    {:[{:[a]}]}
    >>> tc.SFrame.read_csv('interesting_dicts.csv')
    Columns:
      dict  dict
    Rows: 7
    Data:
    +------------------------------+
    |             dict             |
    +------------------------------+
    |  {'dict': 1, 'classic': 1}   |
    | {'separated': 1, 'space': 1} |
    |     {'emptyvalue': None}     |
    |              {}              |
    |         {None: None}         |
    | {'recursive1': [{'a': 'b'}]} |
    | {None: [{None: array('d')}]} |
    +------------------------------+
    [7 rows x 1 columns]

    **Saving**

    Save and load the sframe in native format.

    >>> sf.save('mysframedir')
    >>> sf2 = turicreate.load_sframe('mysframedir')

    **Column Manipulation**

    An SFrame is composed of a collection of columns of SArrays, and individual
    SArrays can be extracted easily. For instance given an SFrame:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    The "id" column can be extracted using:

    >>> sf["id"]
    dtype: int
    Rows: 3
    [1, 2, 3]

    And can be deleted using:

    >>> del sf["id"]

    Multiple columns can be selected by passing a list of column names:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C'],'val2':[5,6,7]})
    >>> sf
    Columns:
        id   int
        val  str
        val2 int
    Rows: 3
    Data:
       id  val val2
    0  1   A   5
    1  2   B   6
    2  3   C   7
    >>> sf2 = sf[['id','val']]
    >>> sf2
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    You can also select columns using types or a list of types:

    >>> sf2 = sf[int]
    >>> sf2
    Columns:
        id   int
        val2 int
    Rows: 3
    Data:
       id  val2
    0  1   5
    1  2   6
    2  3   7

    Or a mix of types and names:

    >>> sf2 = sf[['id', str]]
    >>> sf2
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C


    The same mechanism can be used to re-order columns:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C
    >>> sf[['val','id']]
    >>> sf
    Columns:
        val str
        id  int
    Rows: 3
    Data:
       val id
    0  A   1
    1  B   2
    2  C   3

    **Element Access and Slicing**

    SFrames can be accessed by integer keys just like a regular python list.
    Such operations may not be fast on large datasets so looping over an SFrame
    should be avoided.

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf[0]
    {'id': 1, 'val': 'A'}
    >>> sf[2]
    {'id': 3, 'val': 'C'}
    >>> sf[5]
    IndexError: SFrame index out of range

    Negative indices can be used to access elements from the tail of the array

    >>> sf[-1] # returns the last element
    {'id': 3, 'val': 'C'}
    >>> sf[-2] # returns the second to last element
    {'id': 2, 'val': 'B'}

    The SFrame also supports the full range of python slicing operators:

    >>> sf[1000:] # Returns an SFrame containing rows 1000 to the end
    >>> sf[:1000] # Returns an SFrame containing rows 0 to row 999 inclusive
    >>> sf[0:1000:2] # Returns an SFrame containing rows 0 to row 1000 in steps of 2
    >>> sf[-100:] # Returns an SFrame containing last 100 rows
    >>> sf[-100:len(sf):2] # Returns an SFrame containing last 100 rows in steps of 2

    **Logical Filter**

    An SFrame can be filtered using

    >>> sframe[binary_filter]

    where sframe is an SFrame and binary_filter is an SArray of the same length.
    The result is a new SFrame which contains only rows of the SFrame where its
    matching row in the binary_filter is non zero.

    This permits the use of boolean operators that can be used to perform
    logical filtering operations. For instance, given an SFrame

    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    >>> sf[(sf['id'] >= 1) & (sf['id'] <= 2)]
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B

    See :class:`~turicreate.SArray` for more details on the use of the logical
    filter.

    This can also be used more generally to provide filtering capability which
    is otherwise not expressible with simple boolean functions. For instance:

    >>> sf[sf['id'].apply(lambda x: math.log(x) <= 1)]
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B

    Or alternatively:

    >>> sf[sf.apply(lambda x: math.log(x['id']) <= 1)]

            Create an SFrame from a Python dictionary.

    >>> from turicreate import SFrame
    >>> sf = SFrame({'id':[1,2,3], 'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C
    """

    __slots__ = ['_proxy', '_cache']

    def __init__(self, data=None,
                 format='auto',
                 _proxy=None):
        """__init__(data=list(), format='auto')
        Construct a new SFrame from a url or a pandas.DataFrame.
        """
        # emit metrics for num_rows, num_columns, and type (local://, s3, hdfs, http)

        if (_proxy):
            self.__proxy__ = _proxy
        else:
            self.__proxy__ = UnitySFrameProxy()
            _format = None
            if six.PY2 and isinstance(data, unicode):
                data = data.encode('utf-8')
            if (format == 'auto'):
                if (HAS_PANDAS and isinstance(data, pandas.DataFrame)):
                    _format = 'dataframe'
                elif (isinstance(data, str) or
                      (sys.version_info.major < 3 and isinstance(data, unicode))):

                    if data.endswith(('.csv', '.csv.gz')):
                        _format = 'csv'
                    elif data.endswith(('.tsv', '.tsv.gz')):
                        _format = 'tsv'
                    elif data.endswith(('.txt', '.txt.gz')):
                        print("Assuming file is csv. For other delimiters, " + \
                            "please use `SFrame.read_csv`.")
                        _format = 'csv'
                    else:
                        _format = 'sframe'
                elif type(data) == SArray:
                    _format = 'sarray'

                elif isinstance(data, SFrame):
                    _format = 'sframe_obj'

                elif isinstance(data, dict):
                    _format = 'dict'

                elif _is_non_string_iterable(data):
                    _format = 'array'
                elif data is None:
                    _format = 'empty'
                else:
                    raise ValueError('Cannot infer input type for data ' + str(data))
            else:
                _format = format


            with cython_context():
                if (_format == 'dataframe'):
                    for c in data.columns.values:
                        self.add_column(SArray(data[c].values), str(c), inplace=True)
                elif (_format == 'sframe_obj'):
                    for col in data.column_names():
                        self.__proxy__.add_column(data[col].__proxy__, col)
                elif (_format == 'sarray'):
                    self.__proxy__.add_column(data.__proxy__, '')
                elif (_format == 'array'):
                    if len(data) > 0:
                        unique_types = set([type(x) for x in data if x is not None])
                        if len(unique_types) == 1 and SArray in unique_types:
                            for arr in data:
                                self.add_column(arr, inplace=True)
                        elif SArray in unique_types:
                            raise ValueError("Cannot create SFrame from mix of regular values and SArrays")
                        else:
                            self.__proxy__.add_column(SArray(data).__proxy__, '')
                elif (_format == 'dict'):
                    # Validate that every column is the same length.
                    if len(set(len(value) for value in data.values())) > 1:
                        # probably should be a value error. But we used to raise
                        # runtime error here...
                        raise RuntimeError("All column should be of the same length")
                    # split into SArray values and other iterable values.
                    # We convert the iterable values in bulk, and then add the sarray values as columns
                    sarray_keys = sorted(key for key,value in six.iteritems(data) if isinstance(value, SArray))
                    self.__proxy__.load_from_dataframe({key:value for key,value in six.iteritems(data) if not isinstance(value, SArray)})
                    for key in sarray_keys:
                        self.__proxy__.add_column(data[key].__proxy__, key)
                elif (_format == 'csv'):
                    url = data
                    tmpsf = SFrame.read_csv(url, delimiter=',', header=True)
                    self.__proxy__ = tmpsf.__proxy__
                elif (_format == 'tsv'):
                    url = data
                    tmpsf = SFrame.read_csv(url, delimiter='\t', header=True)
                    self.__proxy__ = tmpsf.__proxy__
                elif (_format == 'sframe'):
                    url = _make_internal_url(data)
                    self.__proxy__.load_from_sframe_index(url)
                elif (_format == 'empty'):
                    pass
                else:
                    raise ValueError('Unknown input type: ' + format)

    @staticmethod
    def _infer_column_types_from_lines(first_rows):
        if (len(first_rows.column_names()) < 1):
          print("Insufficient number of columns to perform type inference")
          raise RuntimeError("Insufficient columns ")
        if len(first_rows) < 1:
          print("Insufficient number of rows to perform type inference")
          raise RuntimeError("Insufficient rows")
        # gets all the values column-wise
        all_column_values_transposed = [list(first_rows[col])
                for col in first_rows.column_names()]
        # transpose
        all_column_values = [list(x) for x in list(zip(*all_column_values_transposed))]
        all_column_type_hints = [[type(t) for t in vals] for vals in all_column_values]
        # collect the hints
        # if every line was inferred to have a different number of elements, die
        if len(set(len(x) for x in all_column_type_hints)) != 1:
            print("Unable to infer column types. Defaulting to str")
            return str

        import types

        column_type_hints = all_column_type_hints[0]
        # now perform type combining across rows
        for i in range(1, len(all_column_type_hints)):
          currow = all_column_type_hints[i]
          for j in range(len(column_type_hints)):
            # combine types
            d = set([currow[j], column_type_hints[j]])
            if (len(d) == 1):
              # easy case. both agree on the type
              continue
            if (((long in d) or (int in d)) and (float in d)):
              # one is an int, one is a float. its a float
              column_type_hints[j] = float
            elif ((array.array in d) and (list in d)):
              # one is an array , one is a list. its a list
              column_type_hints[j] = list
            elif type(None) in d:
              # one is a NoneType. assign to other type
              if currow[j] != type(None):
                  column_type_hints[j] = currow[j]
            else:
              column_type_hints[j] = str
        # final pass. everything which is still NoneType is now a str
        for i in range(len(column_type_hints)):
          if column_type_hints[i] == type(None):
            column_type_hints[i] = str

        return column_type_hints

    @classmethod
    def _read_csv_impl(cls,
                       url,
                       delimiter=',',
                       header=True,
                       error_bad_lines=False,
                       comment_char='',
                       escape_char='\\',
                       double_quote=True,
                       quote_char='\"',
                       skip_initial_space=True,
                       column_type_hints=None,
                       na_values=["NA"],
                       line_terminator="\n",
                       usecols=[],
                       nrows=None,
                       skiprows=0,
                       verbose=True,
                       store_errors=True,
                       nrows_to_infer=100,
                       **kwargs):
        """
        Constructs an SFrame from a CSV file or a path to multiple CSVs, and
        returns a pair containing the SFrame and optionally
        (if store_errors=True) a dict of filenames to SArrays
        indicating for each file, what are the incorrectly parsed lines
        encountered.

        Parameters
        ----------
        store_errors : bool
            If true, the output errors dict will be filled.

        See `read_csv` for the rest of the parameters.
        """
        # Pandas argument compatibility
        if "sep" in kwargs:
            delimiter = kwargs['sep']
            del kwargs['sep']
        if "quotechar" in kwargs:
            quote_char = kwargs['quotechar']
            del kwargs['quotechar']
        if "doublequote" in kwargs:
            double_quote = kwargs['doublequote']
            del kwargs['doublequote']
        if "comment" in kwargs:
            comment_char = kwargs['comment']
            del kwargs['comment']
            if comment_char is None:
                comment_char = ''
        if "lineterminator" in kwargs:
            line_terminator = kwargs['lineterminator']
            del kwargs['lineterminator']
        if len(kwargs) > 0:
            raise TypeError("Unexpected keyword arguments " + str(kwargs.keys()))

        parsing_config = dict()
        parsing_config["delimiter"] = delimiter
        parsing_config["use_header"] = header
        parsing_config["continue_on_failure"] = not error_bad_lines
        parsing_config["comment_char"] = comment_char
        parsing_config["escape_char"] = escape_char
        parsing_config["double_quote"] = double_quote
        parsing_config["quote_char"] = quote_char
        parsing_config["skip_initial_space"] = skip_initial_space
        parsing_config["store_errors"] = store_errors
        parsing_config["line_terminator"] = line_terminator
        parsing_config["output_columns"] = usecols
        parsing_config["skip_rows"] =skiprows

        if type(na_values) is str:
          na_values = [na_values]
        if na_values is not None and len(na_values) > 0:
            parsing_config["na_values"] = na_values

        if nrows is not None:
          parsing_config["row_limit"] = nrows

        proxy = UnitySFrameProxy()
        internal_url = _make_internal_url(url)

        # Attempt to automatically detect the column types. Either produce a
        # list of types; otherwise default to all str types.
        column_type_inference_was_used = False
        if column_type_hints is None:
            try:
                # Get the first nrows_to_infer rows (using all the desired arguments).
                first_rows = SFrame.read_csv(url, nrows=nrows_to_infer,
                                 column_type_hints=type(None),
                                 header=header,
                                 delimiter=delimiter,
                                 comment_char=comment_char,
                                 escape_char=escape_char,
                                 double_quote=double_quote,
                                 quote_char=quote_char,
                                 skip_initial_space=skip_initial_space,
                                 na_values=na_values,
                                 line_terminator=line_terminator,
                                 usecols=usecols,
                                 skiprows=skiprows,
                                 verbose=verbose)
                column_type_hints = SFrame._infer_column_types_from_lines(first_rows)
                typelist = '[' + ','.join(t.__name__ for t in column_type_hints) + ']'
                if verbose:
                    print("------------------------------------------------------")
                    print("Inferred types from first %d line(s) of file as " % nrows_to_infer)
                    print("column_type_hints="+ typelist)
                    print("If parsing fails due to incorrect types, you can correct")
                    print("the inferred type list above and pass it to read_csv in")
                    print( "the column_type_hints argument")
                    print("------------------------------------------------------")
                column_type_inference_was_used = True
            except RuntimeError as e:
                if type(e) == RuntimeError and ("cancel" in str(e.args[0]) or "Cancel" in str(e.args[0])):
                    raise e
                # If the above fails, default back to str for all columns.
                column_type_hints = str
                if verbose:
                    print('Could not detect types. Using str for each column.')

        if type(column_type_hints) is type:
            type_hints = {'__all_columns__': column_type_hints}
        elif type(column_type_hints) is list:
            type_hints = dict(list(zip(['__X%d__' % i for i in range(len(column_type_hints))], column_type_hints)))
        elif type(column_type_hints) is dict:
            # we need to fill in a potentially incomplete dictionary
            try:
                # Get the first nrows_to_infer rows (using all the desired arguments).
                first_rows = SFrame.read_csv(url, nrows=nrows_to_infer,
                                 column_type_hints=type(None),
                                 header=header,
                                 delimiter=delimiter,
                                 comment_char=comment_char,
                                 escape_char=escape_char,
                                 double_quote=double_quote,
                                 quote_char=quote_char,
                                 skip_initial_space=skip_initial_space,
                                 na_values=na_values,
                                 line_terminator=line_terminator,
                                 usecols=usecols,
                                 skiprows=skiprows,
                                 verbose=verbose)
                inferred_types = SFrame._infer_column_types_from_lines(first_rows)
                # make a dict of column_name to type
                inferred_types = dict(list(zip(first_rows.column_names(), inferred_types)))
                # overwrite with the user's specified types
                for key in column_type_hints:
                    inferred_types[key] = column_type_hints[key]
                column_type_hints = inferred_types
            except RuntimeError as e:
                if type(e) == RuntimeError and ("cancel" in e.message or "Cancel" in e.message):
                    raise e
                # If the above fails, default back to str for unmatched columns
                if verbose:
                    print('Could not detect types. Using str for all unspecified columns.')
            type_hints = column_type_hints
        else:
            raise TypeError("Invalid type for column_type_hints. Must be a dictionary, list or a single type.")

        try:
            if (not verbose):
                glconnect.get_server().set_log_progress(False)
            with cython_context():
                errors = proxy.load_from_csvs(internal_url, parsing_config, type_hints)
        except Exception as e:
            if type(e) == RuntimeError and "CSV parsing cancelled" in str(e.args[0]):
                raise e
            if column_type_inference_was_used:
                # try again
                if verbose:
                    print("Unable to parse the file with automatic type inference.")
                    print("Defaulting to column_type_hints=str")
                type_hints = {'__all_columns__': str}
                try:
                    with cython_context():
                        errors = proxy.load_from_csvs(internal_url, parsing_config, type_hints)
                except:
                    glconnect.get_server().set_log_progress(True)
                    raise
            else:
                glconnect.get_server().set_log_progress(True)
                raise

        glconnect.get_server().set_log_progress(True)

        return (cls(_proxy=proxy), { f: SArray(_proxy = es) for (f, es) in errors.items() })

    @classmethod
    def read_csv_with_errors(cls,
                             url,
                             delimiter=',',
                             header=True,
                             comment_char='',
                             escape_char='\\',
                             double_quote=True,
                             quote_char='\"',
                             skip_initial_space=True,
                             column_type_hints=None,
                             na_values=["NA"],
                             line_terminator='\n',
                             usecols = [],
                             nrows=None,
                             skiprows=0,
                             verbose=True,
                             nrows_to_infer=100,
                             **kwargs):
        """
        Constructs an SFrame from a CSV file or a path to multiple CSVs, and
        returns a pair containing the SFrame and a dict of filenames to SArrays
        indicating for each file, what are the incorrectly parsed lines
        encountered.

        Parameters
        ----------
        url : string
            Location of the CSV file or directory to load. If URL is a directory
            or a "glob" pattern, all matching files will be loaded.

        delimiter : string, optional
            This describes the delimiter used for parsing csv files.

        header : bool, optional
            If true, uses the first row as the column names. Otherwise use the
            default column names: 'X1, X2, ...'.

        comment_char : string, optional
            The character which denotes that the
            remainder of the line is a comment.

        escape_char : string, optional
            Character which begins a C escape sequence

        double_quote : bool, optional
            If True, two consecutive quotes in a string are parsed to a single
            quote.

        quote_char : string, optional
            Character sequence that indicates a quote.

        skip_initial_space : bool, optional
            Ignore extra spaces at the start of a field

        column_type_hints : None, type, list[type], dict[string, type], optional
            This provides type hints for each column. By default, this method
            attempts to detect the type of each column automatically.

            Supported types are int, float, str, list, dict, and array.array.

            * If a single type is provided, the type will be
              applied to all columns. For instance, column_type_hints=float
              will force all columns to be parsed as float.
            * If a list of types is provided, the types applies
              to each column in order, e.g.[int, float, str]
              will parse the first column as int, second as float and third as
              string.
            * If a dictionary of column name to type is provided,
              each type value in the dictionary is applied to the key it
              belongs to.
              For instance {'user':int} will hint that the column called "user"
              should be parsed as an integer, and the rest will be type inferred.

        na_values : str | list of str, optional
            A string or list of strings to be interpreted as missing values.

        line_terminator : str, optional
            A string to be interpreted as the line terminator. Defaults to "\\n"
            which will also correctly match Mac, Linux and Windows line endings
            ("\\r", "\\n" and "\\r\\n" respectively)

        usecols : list of str, optional
            A subset of column names to output. If unspecified (default),
            all columns will be read. This can provide performance gains if the
            number of columns are large. If the input file has no headers,
            usecols=['X1','X3'] will read columns 1 and 3.

        nrows : int, optional
            If set, only this many rows will be read from the file.

        skiprows : int, optional
            If set, this number of rows at the start of the file are skipped.

        verbose : bool, optional
            If True, print the progress.

        Returns
        -------
        out : tuple
            The first element is the SFrame with good data. The second element
            is a dictionary of filenames to SArrays indicating for each file,
            what are the incorrectly parsed lines encountered.

        See Also
        --------
        read_csv, SFrame

        Examples
        --------
        >>> bad_url = 'https://static.turi.com/datasets/bad_csv_example.csv'
        >>> (sf, bad_lines) = turicreate.SFrame.read_csv_with_errors(bad_url)
        >>> sf
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [98 rows x 3 columns]

        >>> bad_lines
        {'https://static.turi.com/datasets/bad_csv_example.csv': dtype: str
         Rows: 1
         ['x,y,z,a,b,c']}
       """
        return cls._read_csv_impl(url,
                                  delimiter=delimiter,
                                  header=header,
                                  error_bad_lines=False, # we are storing errors,
                                                         # thus we must not fail
                                                         # on bad lines
                                  comment_char=comment_char,
                                  escape_char=escape_char,
                                  double_quote=double_quote,
                                  quote_char=quote_char,
                                  skip_initial_space=skip_initial_space,
                                  column_type_hints=column_type_hints,
                                  na_values=na_values,
                                  line_terminator=line_terminator,
                                  usecols=usecols,
                                  nrows=nrows,
                                  verbose=verbose,
                                  skiprows=skiprows,
                                  store_errors=True,
                                  nrows_to_infer=nrows_to_infer,
                                  **kwargs)
    @classmethod
    def read_csv(cls,
                 url,
                 delimiter=',',
                 header=True,
                 error_bad_lines=False,
                 comment_char='',
                 escape_char='\\',
                 double_quote=True,
                 quote_char='\"',
                 skip_initial_space=True,
                 column_type_hints=None,
                 na_values=["NA"],
                 line_terminator='\n',
                 usecols=[],
                 nrows=None,
                 skiprows=0,
                 verbose=True,
                 nrows_to_infer=100,
                 **kwargs):
        """
        Constructs an SFrame from a CSV file or a path to multiple CSVs.

        Parameters
        ----------
        url : string
            Location of the CSV file or directory to load. If URL is a directory
            or a "glob" pattern, all matching files will be loaded.

        delimiter : string, optional
            This describes the delimiter used for parsing csv files.

        header : bool, optional
            If true, uses the first row as the column names. Otherwise use the
            default column names : 'X1, X2, ...'.

        error_bad_lines : bool
            If true, will fail upon encountering a bad line. If false, will
            continue parsing skipping lines which fail to parse correctly.
            A sample of the first 10 encountered bad lines will be printed.

        comment_char : string, optional
            The character which denotes that the remainder of the line is a
            comment.

        escape_char : string, optional
            Character which begins a C escape sequence

        double_quote : bool, optional
            If True, two consecutive quotes in a string are parsed to a single
            quote.

        quote_char : string, optional
            Character sequence that indicates a quote.

        skip_initial_space : bool, optional
            Ignore extra spaces at the start of a field

        column_type_hints : None, type, list[type], dict[string, type], optional
            This provides type hints for each column. By default, this method
            attempts to detect the type of each column automatically.

            Supported types are int, float, str, list, dict, and array.array.

            * If a single type is provided, the type will be
              applied to all columns. For instance, column_type_hints=float
              will force all columns to be parsed as float.
            * If a list of types is provided, the types applies
              to each column in order, e.g.[int, float, str]
              will parse the first column as int, second as float and third as
              string.
            * If a dictionary of column name to type is provided,
              each type value in the dictionary is applied to the key it
              belongs to.
              For instance {'user':int} will hint that the column called "user"
              should be parsed as an integer, and the rest will be type inferred.

        na_values : str | list of str, optional
            A string or list of strings to be interpreted as missing values.

        line_terminator : str, optional
            A string to be interpreted as the line terminator. Defaults to "\n"
            which will also correctly match Mac, Linux and Windows line endings
            ("\\r", "\\n" and "\\r\\n" respectively)

        usecols : list of str, optional
            A subset of column names to output. If unspecified (default),
            all columns will be read. This can provide performance gains if the
            number of columns are large. If the input file has no headers,
            usecols=['X1','X3'] will read columns 1 and 3.

        nrows : int, optional
            If set, only this many rows will be read from the file.

        skiprows : int, optional
            If set, this number of rows at the start of the file are skipped.

        verbose : bool, optional
            If True, print the progress.

        Returns
        -------
        out : SFrame

        See Also
        --------
        read_csv_with_errors, SFrame

        Examples
        --------

        Read a regular csv file, with all default options, automatically
        determine types:

        >>> url = 'https://static.turi.com/datasets/rating_data_example.csv'
        >>> sf = turicreate.SFrame.read_csv(url)
        >>> sf
        Columns:
          user_id int
          movie_id  int
          rating  int
        Rows: 10000
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10000 rows x 3 columns]

        Read only the first 100 lines of the csv file:

        >>> sf = turicreate.SFrame.read_csv(url, nrows=100)
        >>> sf
        Columns:
          user_id int
          movie_id  int
          rating  int
        Rows: 100
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [100 rows x 3 columns]

        Read all columns as str type

        >>> sf = turicreate.SFrame.read_csv(url, column_type_hints=str)
        >>> sf
        Columns:
          user_id  str
          movie_id  str
          rating  str
        Rows: 10000
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10000 rows x 3 columns]

        Specify types for a subset of columns and leave the rest to be str.

        >>> sf = turicreate.SFrame.read_csv(url,
        ...                               column_type_hints={
        ...                               'user_id':int, 'rating':float
        ...                               })
        >>> sf
        Columns:
          user_id str
          movie_id  str
          rating  float
        Rows: 10000
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |  3.0   |
        |  25907  |   1663   |  3.0   |
        |  25923  |   1663   |  3.0   |
        |  25924  |   1663   |  3.0   |
        |  25928  |   1663   |  2.0   |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10000 rows x 3 columns]

        Not treat first line as header:

        >>> sf = turicreate.SFrame.read_csv(url, header=False)
        >>> sf
        Columns:
          X1  str
          X2  str
          X3  str
        Rows: 10001
        +---------+----------+--------+
        |    X1   |    X2    |   X3   |
        +---------+----------+--------+
        | user_id | movie_id | rating |
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10001 rows x 3 columns]

        Treat '3' as missing value:

        >>> sf = turicreate.SFrame.read_csv(url, na_values=['3'], column_type_hints=str)
        >>> sf
        Columns:
          user_id str
          movie_id  str
          rating  str
        Rows: 10000
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |  None  |
        |  25907  |   1663   |  None  |
        |  25923  |   1663   |  None  |
        |  25924  |   1663   |  None  |
        |  25928  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10000 rows x 3 columns]

        Throw error on parse failure:

        >>> bad_url = 'https://static.turi.com/datasets/bad_csv_example.csv'
        >>> sf = turicreate.SFrame.read_csv(bad_url, error_bad_lines=True)
        RuntimeError: Runtime Exception. Unable to parse line "x,y,z,a,b,c"
        Set error_bad_lines=False to skip bad lines
        """

        return cls._read_csv_impl(url,
                                  delimiter=delimiter,
                                  header=header,
                                  error_bad_lines=error_bad_lines,
                                  comment_char=comment_char,
                                  escape_char=escape_char,
                                  double_quote=double_quote,
                                  quote_char=quote_char,
                                  skip_initial_space=skip_initial_space,
                                  column_type_hints=column_type_hints,
                                  na_values=na_values,
                                  line_terminator=line_terminator,
                                  usecols=usecols,
                                  nrows=nrows,
                                  skiprows=skiprows,
                                  verbose=verbose,
                                  store_errors=False,
                                  nrows_to_infer=nrows_to_infer,
                                  **kwargs)[0]


    @classmethod
    def read_json(cls,
                  url,
                  orient='records'):
        """
        Reads a JSON file representing a table into an SFrame.

        Parameters
        ----------
        url : string
            Location of the CSV file or directory to load. If URL is a directory
            or a "glob" pattern, all matching files will be loaded.

        orient : string, optional. Either "records" or "lines"
            If orient="records" the file is expected to contain a single JSON
            array, where each array element is a dictionary. If orient="lines",
            the file is expected to contain a JSON element per line.

        Examples
        --------
        The orient parameter describes the expected input format of the JSON
        file.

        If orient="records", the JSON file is expected to contain a single
        JSON Array where each array element is a dictionary describing the row.
        For instance:

        >>> !cat input.json
        [{'a':1,'b':1}, {'a':2,'b':2}, {'a':3,'b':3}]
        >>> SFrame.read_json('input.json', orient='records')
        Columns:
                a	int
                b	int
        Rows: 3
        Data:
        +---+---+
        | a | b |
        +---+---+
        | 1 | 1 |
        | 2 | 2 |
        | 3 | 3 |
        +---+---+

        If orient="lines", the JSON file is expected to contain a JSON element
        per line. If each line contains a dictionary, it is automatically
        unpacked.

        >>> !cat input.json
        {'a':1,'b':1}
        {'a':2,'b':2}
        {'a':3,'b':3}
        >>> g = SFrame.read_json('input.json', orient='lines')
        Columns:
                a	int
                b	int
        Rows: 3
        Data:
        +---+---+
        | a | b |
        +---+---+
        | 1 | 1 |
        | 2 | 2 |
        | 3 | 3 |
        +---+---+

        If the lines are not dictionaries, the original format is maintained.

        >>> !cat input.json
        ['a','b','c']
        ['d','e','f']
        ['g','h','i']
        [1,2,3]
        >>> g = SFrame.read_json('input.json', orient='lines')
        Columns:
                X1	list
        Rows: 3
        Data:
        +-----------+
        |     X1    |
        +-----------+
        | [a, b, c] |
        | [d, e, f] |
        | [g, h, i] |
        +-----------+
        [3 rows x 1 columns]
        """
        if orient == "records":
            g = SArray.read_json(url)
            if len(g) == 0:
                return SFrame()
            g = SFrame({'X1':g})
            return g.unpack('X1','')
        elif orient == "lines":
            g = cls.read_csv(url, header=False)
            if g.num_rows() == 0:
                return SFrame()
            if g.num_columns() != 1:
                raise RuntimeError("Input JSON not of expected format")
            if g['X1'].dtype == dict:
                return g.unpack('X1','')
            else:
                return g
        else:
            raise ValueError("Invalid value for orient parameter (" + str(orient) + ")")


    @classmethod
    def from_sql(cls, conn, sql_statement, params=None, type_inference_rows=100,
        dbapi_module=None, column_type_hints=None, cursor_arraysize=128):
        """
        Convert the result of a SQL database query to an SFrame.

        Parameters
        ----------
        conn : dbapi2.Connection
          A DBAPI2 connection object. Any connection object originating from
          the 'connect' method of a DBAPI2-compliant package can be used.

        sql_statement : str
          The query to be sent to the database through the given connection.
          No checks are performed on the `sql_statement`. Any side effects from
          the query will be reflected on the database.  If no result rows are
          returned, an empty SFrame is created.

        params : iterable | dict, optional
          Parameters to substitute for any parameter markers in the
          `sql_statement`. Be aware that the style of parameters may vary
          between different DBAPI2 packages.

        type_inference_rows : int, optional
          The maximum number of rows to use for determining the column types of
          the SFrame. These rows are held in Python until all column types are
          determined or the maximum is reached.

        dbapi_module : module | package, optional
          The top-level DBAPI2 module/package that constructed the given
          connection object. By default, a best guess of which module the
          connection came from is made. In the event that this guess is wrong,
          this will need to be specified.

        column_type_hints : dict | list | type, optional
          Specifies the types of the output SFrame. If a dict is given, it must
          have result column names as keys, but need not have all of the result
          column names. If a list is given, the length of the list must match
          the number of result columns. If a single type is given, all columns
          in the output SFrame will be this type. If the result type is
          incompatible with the types given in this argument, a casting error
          will occur.

        cursor_arraysize : int, optional
          The number of rows to fetch from the database at one time.

        Returns
        -------
        out : SFrame

        Examples
        --------
        >>> import sqlite3

        >>> conn = sqlite3.connect('example.db')

        >>> turicreate.SFrame.from_sql(conn, "SELECT * FROM foo")
        Columns:
                a       int
                b       int
        Rows: 1
        Data:
        +---+---+
        | a | b |
        +---+---+
        | 1 | 2 |
        +---+---+
        [1 rows x 2 columns]
        """
        # Mapping types is always the trickiest part about reading from a
        # database, so the main complexity of this function concerns types.
        # Much of the heavy-lifting of this is done by the DBAPI2 module, which
        # holds the burden of the actual mapping from the database-specific
        # type to a suitable Python type. The problem is that the type that the
        # module chooses may not be supported by SFrame, and SFrame needs a
        # list of types to be created, so we must resort to guessing the type
        # of a column if the query result returns lots of NULL values. The goal
        # of these steps is to fail as little as possible first, and then
        # preserve data as much as we can.
        #
        # Here is how the type for an SFrame column is chosen:
        #
        # 1. The column_type_hints parameter is checked.
        #
        #    Each column specified in the parameter will be forced to the
        #    hinted type via a Python-side cast before it is given to the
        #    SFrame. Only int, float, and str are allowed to be hints.
        #
        # 2. The types returned from the cursor are checked.
        #
        #    The first non-None result for each column is taken to be the type
        #    of that column. The type is checked for whether SFrame supports
        #    it, or whether it can convert to a supported type. If the type is
        #    supported, no Python-side cast takes place. If unsupported, the
        #    SFrame column is set to str and the values are casted in Python to
        #    str before being added to the SFrame.
        #
        # 3. DB type codes provided by module are checked
        #
        #    This case happens for any column that only had None values in the
        #    first `type_inference_rows` rows. In this case we check the
        #    type_code in the cursor description for the columns missing types.
        #    These types often do not match up with an SFrame-supported Python
        #    type, so the utility of this step is limited. It can only result
        #    in labeling datetime.datetime, float, or str. If a suitable
        #    mapping isn't found, we fall back to str.
        mod_info = _get_global_dbapi_info(dbapi_module, conn)

        from .sframe_builder import SFrameBuilder

        c = conn.cursor()
        try:
            if params is None:
                c.execute(sql_statement)
            else:
                c.execute(sql_statement, params)
        except mod_info['Error'] as e:
            # The rollback method is considered optional by DBAPI2, but some
            # modules that do implement it won't work again unless it is called
            # if an error happens on a cursor.
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise e

        c.arraysize = cursor_arraysize

        result_desc = c.description
        result_names = [i[0] for i in result_desc]
        result_types = [None for i in result_desc]

        cols_to_force_cast = set()
        temp_vals = []

        # Set any types that are given to us
        col_name_to_num = {result_names[i]:i for i in range(len(result_names))}
        if column_type_hints is not None:
            if type(column_type_hints) is dict:
                for k,v in column_type_hints.items():
                    col_num = col_name_to_num[k]
                    cols_to_force_cast.add(col_num)
                    result_types[col_num] = v
            elif type(column_type_hints) is list:
                if len(column_type_hints) != len(result_names):
                    __LOGGER__.warn("If column_type_hints is specified as a "+\
                        "list, it must be of the same size as the result "+\
                        "set's number of columns. Ignoring (use dict instead).")
                else:
                    result_types = column_type_hints
                    cols_to_force_cast.update(range(len(result_desc)))
            elif type(column_type_hints) is type:
                result_types = [column_type_hints for i in result_desc]
                cols_to_force_cast.update(range(len(result_desc)))

        # Since we will be casting whatever we receive to the types given
        # before submitting the values to the SFrame, we need to make sure that
        # these are types that a "cast" makes sense, and we're not calling a
        # constructor that expects certain input (e.g.  datetime.datetime),
        # since we could get lots of different input
        hintable_types = [int,float,str]
        if not all([i in hintable_types or i is None for i in result_types]):
            raise TypeError("Only " + str(hintable_types) + " can be provided as type hints!")

        # Perform type inference by checking to see what python types are
        # returned from the cursor
        if not all(result_types):
            # Only test the first fetch{one,many} command since the only way it
            # will raise an exception is if execute didn't produce a result set
            try:
                row = c.fetchone()
            except mod_info['Error'] as e:
                if hasattr(conn, 'rollback'):
                    conn.rollback()
                raise e
            while row is not None:
                # Assumes that things like dicts are not a "single sequence"
                temp_vals.append(row)
                val_count = 0
                for val in row:
                    if result_types[val_count] is None and val is not None:
                        result_types[val_count] = type(val)
                    val_count += 1
                if all(result_types) or len(temp_vals) >= type_inference_rows:
                    break
                row = c.fetchone()

        # This will be true if some columns have all missing values up to this
        # point. Try using DBAPI2 type_codes to pick a suitable type. If this
        # doesn't work, fall back to string.
        if not all(result_types):
            missing_val_cols = [i for i,v in enumerate(result_types) if v is None]
            cols_to_force_cast.update(missing_val_cols)
            inferred_types = infer_dbapi2_types(c, mod_info)
            cnt = 0
            for i in result_types:
                if i is None:
                    result_types[cnt] = inferred_types[cnt]
                cnt += 1

        sb = SFrameBuilder(result_types, column_names=result_names)
        unsupported_cols = [i for i,v in enumerate(sb.column_types()) if v is type(None)]
        if len(unsupported_cols) > 0:
            cols_to_force_cast.update(unsupported_cols)
            for i in unsupported_cols:
                result_types[i] = str
            sb = SFrameBuilder(result_types, column_names=result_names)

        temp_vals = _convert_rows_to_builtin_seq(temp_vals)
        sb.append_multiple(_force_cast_sql_types(temp_vals, result_types, cols_to_force_cast))
        rows = c.fetchmany()
        while len(rows) > 0:
            rows = _convert_rows_to_builtin_seq(rows)
            sb.append_multiple(_force_cast_sql_types(rows, result_types, cols_to_force_cast))
            rows = c.fetchmany()
        cls = sb.close()

        try:
            c.close()
        except mod_info['Error'] as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise e
        return cls

    def to_sql(self, conn, table_name, dbapi_module=None,
            use_python_type_specifiers=False, use_exact_column_names=True):
        """
        Convert an SFrame to a single table in a SQL database.

        This function does not attempt to create the table or check if a table
        named `table_name` exists in the database. It simply assumes that
        `table_name` exists in the database and appends to it.

        `to_sql` can be thought of as a convenience wrapper around
        parameterized SQL insert statements.

        Parameters
        ----------
        conn : dbapi2.Connection
          A DBAPI2 connection object. Any connection object originating from
          the 'connect' method of a DBAPI2-compliant package can be used.

        table_name : str
          The name of the table to append the data in this SFrame.

        dbapi_module : module | package, optional
          The top-level DBAPI2 module/package that constructed the given
          connection object. By default, a best guess of which module the
          connection came from is made. In the event that this guess is wrong,
          this will need to be specified.

        use_python_type_specifiers : bool, optional
          If the DBAPI2 module's parameter marker style is 'format' or
          'pyformat', attempt to use accurate type specifiers for each value
          ('s' for string, 'd' for integer, etc.). Many DBAPI2 modules simply
          use 's' for all types if they use these parameter markers, so this is
          False by default.

        use_exact_column_names : bool, optional
          Specify the column names of the SFrame when inserting its contents
          into the DB. If the specified table does not have the exact same
          column names as the SFrame, inserting the data will fail. If False,
          the columns in the SFrame are inserted in order without care of the
          schema of the DB table. True by default.
        """
        mod_info = _get_global_dbapi_info(dbapi_module, conn)
        c = conn.cursor()

        col_info = list(zip(self.column_names(), self.column_types()))

        if not use_python_type_specifiers:
            pytype_to_printf = lambda x: 's'

        # DBAPI2 standard allows for five different ways to specify parameters
        sql_param = {
            'qmark'   : lambda name,col_num,col_type: '?',
            'numeric' : lambda name,col_num,col_type:':'+str(col_num+1),
            'named'   : lambda name,col_num,col_type:':'+str(name),
            'format'  : lambda name,col_num,col_type:'%'+pytype_to_printf(col_type),
            'pyformat': lambda name,col_num,col_type:'%('+str(name)+')'+pytype_to_printf(col_type),
            }

        get_sql_param = sql_param[mod_info['paramstyle']]

        # form insert string
        ins_str = "INSERT INTO " + str(table_name)
        value_str = " VALUES ("
        col_str = " ("
        count = 0
        for i in col_info:
            col_str += i[0]
            value_str += get_sql_param(i[0],count,i[1])
            if count < len(col_info)-1:
                col_str += ","
                value_str += ","
            count += 1
        col_str += ")"
        value_str += ")"

        if use_exact_column_names:
            ins_str += col_str

        ins_str += value_str

        # Some formats require values in an iterable, some a dictionary
        if (mod_info['paramstyle'] == 'named' or\
            mod_info['paramstyle'] == 'pyformat'):
          prepare_sf_row = lambda x:x
        else:
          col_names = self.column_names()
          prepare_sf_row = lambda x: [x[i] for i in col_names]

        for i in self:
            try:
                c.execute(ins_str, prepare_sf_row(i))
            except mod_info['Error'] as e:
                if hasattr(conn, 'rollback'):
                    conn.rollback()
                raise e

        conn.commit()
        c.close()


    def __hash__(self):
        '''
        Because we override `__eq__` we need to implement this function in Python 3.
        Just make it match default behavior in Python 2.
        '''
        return id(self) // 16


    def __repr__(self):
        """
        Returns a string description of the frame
        """
        ret = self.__get_column_description__()
        (is_empty, data_str) = self.__str_impl__()
        if is_empty:
            data_str = "\t[]"

        if self.__has_size__():
            ret = ret + "Rows: " + str(len(self)) + "\n\n"
        else:
            ret = ret + "Rows: Unknown" + "\n\n"

        ret = ret + "Data:\n"
        ret = ret + data_str
        return ret

    def __get_column_description__(self):
        colnames = self.column_names()
        coltypes = self.column_types()
        ret = "Columns:\n"
        if len(colnames) > 0:
            for i in range(len(colnames)):
                ret = ret + "\t" + colnames[i] + "\t" + coltypes[i].__name__ + "\n"
            ret = ret + "\n"
        else:
            ret = ret + "\tNone\n\n"
        return ret

    def __get_pretty_tables__(self, wrap_text=False, max_row_width=80,
                              max_column_width=30, max_columns=20,
                              max_rows_to_display=60):
        """
        Returns a list of pretty print tables representing the current SFrame.
        If the number of columns is larger than max_columns, the last pretty
        table will contain an extra column of "...".
        Parameters
        ----------
        wrap_text : bool, optional
        max_row_width : int, optional
            Max number of characters per table.
        max_column_width : int, optional
            Max number of characters per column.
        max_columns : int, optional
            Max number of columns per table.
        max_rows_to_display : int, optional
            Max number of rows to display.
        Returns
        -------
        out : list[PrettyTable]
        """
        if (len(self) <= max_rows_to_display):
            headsf = self.__copy__()
        else:
            headsf = self.head(max_rows_to_display)

        if headsf.shape == (0, 0):
            return [PrettyTable()]

        # convert array.array column to list column so they print like [...]
        # and not array('d', ...)
        for col in headsf.column_names():
            if headsf[col].dtype is array.array:
                headsf[col] = headsf[col].astype(list)

        def _value_to_str(value):
            if (type(value) is array.array):
                return str(list(value))
            elif (type(value) is numpy.ndarray):
                return str(value).replace('\n',' ')
            elif (type(value) is list):
                return '[' + ", ".join(_value_to_str(x) for x in value) + ']'
            else:
                return str(value)

        def _escape_space(s):
            if sys.version_info.major == 3:
                return "".join([ch.encode('unicode_escape').decode() if ch.isspace() else ch for ch in s])
            return "".join([ch.encode('string_escape') if ch.isspace() else ch for ch in s])

        def _truncate_respect_unicode(s, max_length):
            if (len(s) <= max_length):
                return s
            else:
                if sys.version_info.major < 3:
                    u = unicode(s, 'utf-8', errors='replace')
                    return u[:max_length].encode('utf-8')
                else:
                    return s[:max_length]


        def _truncate_str(s, wrap_str=False):
            """
            Truncate and optionally wrap the input string as unicode, replace
            unconvertible character with a diamond ?.
            """
            s = _escape_space(s)

            if len(s) <= max_column_width:
                if sys.version_info.major < 3:
                    return unicode(s, 'utf-8', errors='replace')
                else:
                    return s
            else:
                ret = ''
                # if wrap_str is true, wrap the text and take at most 2 rows
                if wrap_str:
                    wrapped_lines = wrap(s, max_column_width)
                    if len(wrapped_lines) == 1:
                        return wrapped_lines[0]
                    last_line = wrapped_lines[1]
                    if len(last_line) >= max_column_width:
                        last_line = _truncate_respect_unicode(last_line, max_column_width - 4)
                    ret = wrapped_lines[0] + '\n' + last_line + ' ...'
                else:
                    ret = _truncate_respect_unicode(s, max_column_width - 4) + '...'

                if sys.version_info.major < 3:
                    return unicode(ret, 'utf-8', errors='replace')
                else:
                    return ret

        columns = self.column_names()[:max_columns]
        columns.reverse()  # reverse the order of columns and we will pop from the end

        num_column_of_last_table = 0
        row_of_tables = []
        # let's build a list of tables with max_columns
        # each table should satisfy, max_row_width, and max_column_width
        while len(columns) > 0:
            tbl = PrettyTable()
            table_width = 0
            num_column_of_last_table = 0
            while len(columns) > 0:
                col = columns.pop()
                # check the max length of element in the column
                if len(headsf) > 0:
                    col_width = min(max_column_width, max(len(str(x)) for x in headsf[col]))
                else:
                    col_width = max_column_width
                if (table_width + col_width < max_row_width):
                    # truncate the header if necessary
                    header = _truncate_str(col, wrap_text)
                    tbl.add_column(header, [_truncate_str(_value_to_str(x), wrap_text) for x in headsf[col]])
                    table_width = str(tbl).find('\n')
                    num_column_of_last_table += 1
                else:
                    # the column does not fit in the current table, push it back to columns
                    columns.append(col)
                    break
            tbl.align = 'c'
            row_of_tables.append(tbl)

        # add a column of all "..." if there are more columns than displayed
        if self.num_columns() > max_columns:
            row_of_tables[-1].add_column('...', ['...'] * len(headsf))
            num_column_of_last_table += 1

        # add a row of all "..." if there are more rows than displayed
        if self.__has_size__() and self.num_rows() > headsf.num_rows():
            row_of_tables[-1].add_row(['...'] * num_column_of_last_table)
        return row_of_tables

    def print_rows(self, num_rows=10, num_columns=40, max_column_width=30,
                   max_row_width=80, output_file=None):
        """
        Print the first M rows and N columns of the SFrame in human readable
        format.

        Parameters
        ----------
        num_rows : int, optional
            Number of rows to print.

        num_columns : int, optional
            Number of columns to print.

        max_column_width : int, optional
            Maximum width of a column. Columns use fewer characters if possible.

        max_row_width : int, optional
            Maximum width of a printed row. Columns beyond this width wrap to a
            new line. `max_row_width` is automatically reset to be the
            larger of itself and `max_column_width`.

        output_file: file, optional
            The stream or file that receives the output. By default the output
            goes to sys.stdout, but it can also be redirected to a file or a
            string (using an object of type StringIO).

        See Also
        --------
        head, tail
        """
        if output_file is None:
            output_file = sys.stdout

        max_row_width = max(max_row_width, max_column_width + 1)

        printed_sf = self._imagecols_to_stringcols(num_rows)
        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=False,
                                                         max_rows_to_display=num_rows,
                                                         max_columns=num_columns,
                                                         max_column_width=max_column_width,
                                                         max_row_width=max_row_width)
        footer = "[%d rows x %d columns]\n" % self.shape
        print('\n'.join([str(tb) for tb in row_of_tables]) + "\n" + footer, file=output_file)

    def _imagecols_to_stringcols(self, num_rows=10):
        # A list of column types
        types = self.column_types()
        # A list of indexable column names
        names = self.column_names()

        # Constructing names of sframe columns that are of image type
        image_column_names = [names[i] for i in range(len(names)) if types[i] == _Image]

        #If there are image-type columns, copy the SFrame and cast the top MAX_NUM_ROWS_TO_DISPLAY of those columns to string
        printed_sf = self.__copy__()
        if len(image_column_names) > 0:
            for t in names:
                if t in image_column_names:
                    printed_sf[t] = self[t].astype(str)
        return printed_sf.head(num_rows)

    def __str_impl__(self, num_rows=10, footer=True):
        """
        Returns a string containing the first num_rows elements of the frame, along
        with a description of the frame.
        """
        MAX_ROWS_TO_DISPLAY = num_rows

        printed_sf = self._imagecols_to_stringcols(MAX_ROWS_TO_DISPLAY)

        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=False, max_rows_to_display=MAX_ROWS_TO_DISPLAY)
        is_empty = len(printed_sf) == 0

        if (not footer):
            return (is_empty, '\n'.join([str(tb) for tb in row_of_tables]))

        if self.__has_size__():
            footer = '[%d rows x %d columns]\n' % self.shape
            if (self.num_rows() > MAX_ROWS_TO_DISPLAY):
                footer += '\n'.join(FOOTER_STRS)
        else:
            footer = '[? rows x %d columns]\n' % self.num_columns()
            footer += '\n'.join(LAZY_FOOTER_STRS)
        return (is_empty, '\n'.join([str(tb) for tb in row_of_tables]) + "\n" + footer)

    def __str__(self, num_rows=10, footer=True):
        """
        Returns a string containing the first 10 elements of the frame, along
        with a description of the frame.
        """
        return self.__str_impl__(num_rows, footer)[1]

    def _repr_html_(self):
        MAX_ROWS_TO_DISPLAY = 10

        printed_sf = self._imagecols_to_stringcols(MAX_ROWS_TO_DISPLAY)

        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=True,
                                                         max_row_width=120,
                                                         max_columns=40,
                                                         max_column_width=25,
                                                         max_rows_to_display=MAX_ROWS_TO_DISPLAY)
        if self.__has_size__():
            footer = '[%d rows x %d columns]<br/>' % self.shape
            if (self.num_rows() > MAX_ROWS_TO_DISPLAY):
                footer += '<br/>'.join(FOOTER_STRS)
        else:
            footer = '[? rows x %d columns]<br/>' % self.num_columns()
            footer += '<br/>'.join(LAZY_FOOTER_STRS)
        begin = '<div style="max-height:1000px;max-width:1500px;overflow:auto;">'
        end = '\n</div>'
        return begin + '\n'.join([tb.get_html_string(format=True) for tb in row_of_tables]) + "\n" + footer + end

    def __nonzero__(self):
        """
        Returns true if the frame is not empty.
        """
        return self.num_rows() != 0

    def __len__(self):
        """
        Returns the number of rows of the sframe.
        """
        return self.num_rows()

    def __copy__(self):
        """
        Returns a shallow copy of the sframe.
        """
        return self.select_columns(self.column_names())

    def __deepcopy__(self, memo):
        """
        Returns a deep copy of the sframe. As the data in an SFrame is
        immutable, this is identical to __copy__.
        """
        return self.__copy__()

    def copy(self):
        """
        Returns a shallow copy of the sframe.
        """
        return self.__copy__()

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def _row_selector(self, other):
        """
        Where other is an SArray of identical length as the current Frame,
        this returns a selection of a subset of rows in the current SFrame
        where the corresponding row in the selector is non-zero.
        """
        if type(other) is SArray:
            if self.__has_size__() and other.__has_size__() and len(other) != len(self):
                raise IndexError("Cannot perform logical indexing on arrays of different length.")
            with cython_context():
                return SFrame(_proxy=self.__proxy__.logical_filter(other.__proxy__))

    @property
    def dtype(self):
        """
        The type of each column.

        Returns
        -------
        out : list[type]
            Column types of the SFrame.

        See Also
        --------
        column_types
        """
        return self.column_types()

    def num_rows(self):
        """
        The number of rows in this SFrame.

        Returns
        -------
        out : int
            Number of rows in the SFrame.

        See Also
        --------
        num_columns
        """
        return self.__proxy__.num_rows()

    def num_columns(self):
        """
        The number of columns in this SFrame.

        Returns
        -------
        out : int
            Number of columns in the SFrame.

        See Also
        --------
        num_rows
        """
        return self.__proxy__.num_columns()

    def column_names(self):
        """
        The name of each column in the SFrame.

        Returns
        -------
        out : list[string]
            Column names of the SFrame.

        See Also
        --------
        rename
        """
        return self.__proxy__.column_names()

    def column_types(self):
        """
        The type of each column in the SFrame.

        Returns
        -------
        out : list[type]
            Column types of the SFrame.

        See Also
        --------
        dtype
        """
        return self.__proxy__.dtype()

    def head(self, n=10):
        """
        The first n rows of the SFrame.

        Parameters
        ----------
        n : int, optional
            The number of rows to fetch.

        Returns
        -------
        out : SFrame
            A new SFrame which contains the first n rows of the current SFrame

        See Also
        --------
        tail, print_rows
        """
        return SFrame(_proxy=self.__proxy__.head(n))

    def to_dataframe(self):
        """
        Convert this SFrame to pandas.DataFrame.

        This operation will construct a pandas.DataFrame in memory. Care must
        be taken when size of the returned object is big.

        Returns
        -------
        out : pandas.DataFrame
            The dataframe which contains all rows of SFrame
        """
        assert HAS_PANDAS, 'pandas is not installed.'
        df = pandas.DataFrame()
        for i in range(self.num_columns()):
            column_name = self.column_names()[i]
            df[column_name] = list(self[column_name])
            if len(df[column_name]) == 0:
                df[column_name] = df[column_name].astype(self.column_types()[i])
        return df

    def to_numpy(self):
        """
        Converts this SFrame to a numpy array

        This operation will construct a numpy array in memory. Care must
        be taken when size of the returned object is big.

        Returns
        -------
        out : numpy.ndarray
            A Numpy Array containing all the values of the SFrame

        """
        assert HAS_NUMPY, 'numpy is not installed.'
        import numpy
        return numpy.transpose(numpy.asarray([self[x] for x in self.column_names()]))

    def tail(self, n=10):
        """
        The last n rows of the SFrame.

        Parameters
        ----------
        n : int, optional
            The number of rows to fetch.

        Returns
        -------
        out : SFrame
            A new SFrame which contains the last n rows of the current SFrame

        See Also
        --------
        head, print_rows
        """
        return SFrame(_proxy=self.__proxy__.tail(n))

    def apply(self, fn, dtype=None, seed=None):
        """
        Transform each row to an :class:`~turicreate.SArray` according to a
        specified function. Returns a new SArray of ``dtype`` where each element
        in this SArray is transformed by `fn(x)` where `x` is a single row in
        the sframe represented as a dictionary.  The ``fn`` should return
        exactly one value which can be cast into type ``dtype``. If ``dtype`` is
        not specified, the first 100 rows of the SFrame are used to make a guess
        of the target data type.

        Parameters
        ----------
        fn : function
            The function to transform each row of the SFrame. The return
            type should be convertible to `dtype` if `dtype` is not None.
            This can also be a toolkit extension function which is compiled
            as a native shared library using SDK.

        dtype : dtype, optional
            The dtype of the new SArray. If None, the first 100
            elements of the array are used to guess the target
            data type.

        seed : int, optional
            Used as the seed if a random number generator is included in `fn`.

        Returns
        -------
        out : SArray
            The SArray transformed by fn.  Each element of the SArray is of
            type ``dtype``

        Examples
        --------
        Concatenate strings from several columns:

        >>> sf = turicreate.SFrame({'user_id': [1, 2, 3], 'movie_id': [3, 3, 6],
                                  'rating': [4, 5, 1]})
        >>> sf.apply(lambda x: str(x['user_id']) + str(x['movie_id']) + str(x['rating']))
        dtype: str
        Rows: 3
        ['134', '235', '361']
        """
        assert callable(fn), "Input must be callable"
        test_sf = self[:10]
        dryrun = [fn(row) for row in test_sf]
        if dtype is None:
            dtype = SArray(dryrun).dtype

        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)


        nativefn = None
        try:
            from .. import extensions as extensions
            nativefn = extensions._build_native_function_call(fn)
        except:
            pass

        if nativefn is not None:
            # this is a toolkit lambda. We can do something about it
            with cython_context():
                return SArray(_proxy=self.__proxy__.transform_native(nativefn, dtype, seed))

        with cython_context():
            return SArray(_proxy=self.__proxy__.transform(fn, dtype, seed))

    def flat_map(self, column_names, fn, column_types='auto', seed=None):
        """
        Map each row of the SFrame to multiple rows in a new SFrame via a
        function.

        The output of `fn` must have type List[List[...]].  Each inner list
        will be a single row in the new output, and the collection of these
        rows within the outer list make up the data for the output SFrame.
        All rows must have the same length and the same order of types to
        make sure the result columns are homogeneously typed.  For example, if
        the first element emitted into in the outer list by `fn` is
        [43, 2.3, 'string'], then all other elements emitted into the outer
        list must be a list with three elements, where the first is an int,
        second is a float, and third is a string.  If column_types is not
        specified, the first 10 rows of the SFrame are used to determine the
        column types of the returned sframe.

        Parameters
        ----------
        column_names : list[str]
            The column names for the returned SFrame.

        fn : function
            The function that maps each of the sframe row into multiple rows,
            returning List[List[...]].  All outputted rows must have the same
            length and order of types.

        column_types : list[type], optional
            The column types of the output SFrame. Default value will be
            automatically inferred by running `fn` on the first 10 rows of the
            input. If the types cannot be inferred from the first 10 rows, an
            error is raised.

        seed : int, optional
            Used as the seed if a random number generator is included in `fn`.

        Returns
        -------
        out : SFrame
            A new SFrame containing the results of the flat_map of the
            original SFrame.

        Examples
        ---------
        Repeat each row according to the value in the 'number' column.

        >>> sf = turicreate.SFrame({'letter': ['a', 'b', 'c'],
        ...                       'number': [1, 2, 3]})
        >>> sf.flat_map(['number', 'letter'],
        ...             lambda x: [list(x.itervalues()) for i in range(0, x['number'])])
        +--------+--------+
        | number | letter |
        +--------+--------+
        |   1    |   a    |
        |   2    |   b    |
        |   2    |   b    |
        |   3    |   c    |
        |   3    |   c    |
        |   3    |   c    |
        +--------+--------+
        [6 rows x 2 columns]
        """
        assert callable(fn), "Input must be callable"
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)


        # determine the column_types
        if column_types == 'auto':
            types = set()
            sample = self[0:10]
            results = [fn(row) for row in sample]

            for rows in results:
                if type(rows) is not list:
                    raise TypeError("Output type of the lambda function must be a list of lists")

                # note: this skips empty lists
                for row in rows:
                    if type(row) is not list:
                        raise TypeError("Output type of the lambda function must be a list of lists")
                    types.add(tuple([type(v) for v in row]))

            if len(types) == 0:
                raise TypeError(
                    "Could not infer output column types from the first ten rows " +\
                    "of the SFrame. Please use the 'column_types' parameter to " +\
                    "set the types.")

            if len(types) > 1:
                raise TypeError("Mapped rows must have the same length and types")

            column_types = list(types.pop())

        assert type(column_types) is list, "'column_types' must be a list."
        assert len(column_types) == len(column_names), "Number of output columns must match the size of column names"
        with cython_context():
            return SFrame(_proxy=self.__proxy__.flat_map(fn, column_names, column_types, seed))

    def sample(self, fraction, seed=None, exact=False):
        """
        Sample a fraction of the current SFrame's rows.

        Parameters
        ----------
        fraction : float
            Fraction of the rows to fetch. Must be between 0 and 1.
            if exact is False (default), the number of rows returned is
            approximately the fraction times the number of rows.

        seed : int, optional
            Seed for the random number generator used to sample.

        exact: bool, optional
            Defaults to False. If exact=True, an exact fraction is returned, 
            but at a performance penalty.

        Returns
        -------
        out : SFrame
            A new SFrame containing sampled rows of the current SFrame.

        Examples
        --------
        Suppose we have an SFrame with 6,145 rows.

        >>> import random
        >>> sf = SFrame({'id': range(0, 6145)})

        Retrieve about 30% of the SFrame rows with repeatable results by
        setting the random seed.

        >>> len(sf.sample(.3, seed=5))
        1783
        """
        if seed is None:
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)

        if (fraction > 1 or fraction < 0):
            raise ValueError('Invalid sampling rate: ' + str(fraction))


        if (self.num_rows() == 0 or self.num_columns() == 0):
            return self
        else:
            with cython_context():
                return SFrame(_proxy=self.__proxy__.sample(fraction, seed, exact))

    def random_split(self, fraction, seed=None, exact=False):
        """
        Randomly split the rows of an SFrame into two SFrames. The first SFrame
        contains *M* rows, sampled uniformly (without replacement) from the
        original SFrame. *M* is approximately the fraction times the original
        number of rows. The second SFrame contains the remaining rows of the
        original SFrame. 
        
        An exact fraction partition can be optionally obtained by setting 
        exact=True.

        Parameters
        ----------
        fraction : float
            Fraction of the rows to fetch. Must be between 0 and 1.
            if exact is False (default), the number of rows returned is
            approximately the fraction times the number of rows.

        seed : int, optional
            Seed for the random number generator used to split.

        exact: bool, optional
            Defaults to False. If exact=True, an exact fraction is returned, 
            but at a performance penalty.

        Returns
        -------
        out : tuple [SFrame]
            Two new SFrames.

        Examples
        --------
        Suppose we have an SFrame with 1,024 rows and we want to randomly split
        it into training and testing datasets with about a 90%/10% split.

        >>> sf = turicreate.SFrame({'id': range(1024)})
        >>> sf_train, sf_test = sf.random_split(.9, seed=5)
        >>> print(len(sf_train), len(sf_test))
        922 102
        """
        if (fraction > 1 or fraction < 0):
            raise ValueError('Invalid sampling rate: ' + str(fraction))
        if (self.num_rows() == 0 or self.num_columns() == 0):
            return (SFrame(), SFrame())

        if seed is None:
            # Include the nanosecond component as well.
            seed = abs(hash("%0.20f" % time.time())) % (2 ** 31)

        # The server side requires this to be an int, so cast if we can
        try:
            seed = int(seed)
        except ValueError:
            raise ValueError('The \'seed\' parameter must be of type int.')


        with cython_context():
            proxy_pair = self.__proxy__.random_split(fraction, seed, exact)
            return (SFrame(data=[], _proxy=proxy_pair[0]), SFrame(data=[], _proxy=proxy_pair[1]))

    def topk(self, column_name, k=10, reverse=False):
        """
        Get top k rows according to the given column. Result is according to and
        sorted by `column_name` in the given order (default is descending).
        When `k` is small, `topk` is more efficient than `sort`.

        Parameters
        ----------
        column_name : string
            The column to sort on

        k : int, optional
            The number of rows to return

        reverse : bool, optional
            If True, return the top k rows in ascending order, otherwise, in
            descending order.

        Returns
        -------
        out : SFrame
            an SFrame containing the top k rows sorted by column_name.

        See Also
        --------
        sort

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': range(1000)})
        >>> sf['value'] = -sf['id']
        >>> sf.topk('id', k=3)
        +--------+--------+
        |   id   |  value |
        +--------+--------+
        |   999  |  -999  |
        |   998  |  -998  |
        |   997  |  -997  |
        +--------+--------+
        [3 rows x 2 columns]

        >>> sf.topk('value', k=3)
        +--------+--------+
        |   id   |  value |
        +--------+--------+
        |   1    |  -1    |
        |   2    |  -2    |
        |   3    |  -3    |
        +--------+--------+
        [3 rows x 2 columns]
        """
        if type(column_name) is not str:
            raise TypeError("column_name must be a string")


        sf = self[self[column_name].is_topk(k, reverse)]
        return sf.sort(column_name, ascending=reverse)

    def save(self, filename, format=None):
        """
        Save the SFrame to a file system for later use.

        Parameters
        ----------
        filename : string
            The location to save the SFrame. Either a local directory or a
            remote URL. If the format is 'binary', a directory will be created
            at the location which will contain the sframe.

        format : {'binary', 'csv', 'json'}, optional
            Format in which to save the SFrame. Binary saved SFrames can be
            loaded much faster and without any format conversion losses. If not
            given, will try to infer the format from filename given. If file
            name ends with 'csv' or '.csv.gz', then save as 'csv' format,
            otherwise save as 'binary' format.
            See export_csv for more csv saving options.

        See Also
        --------
        load_sframe, SFrame

        Examples
        --------
        >>> # Save the sframe into binary format
        >>> sf.save('data/training_data_sframe')

        >>> # Save the sframe into csv format
        >>> sf.save('data/training_data.csv', format='csv')
        """

        if format is None:
            if filename.endswith(('.csv', '.csv.gz')):
                format = 'csv'
            else:
                format = 'binary'
        else:
            if format is 'csv':
                if not filename.endswith(('.csv', '.csv.gz')):
                    filename = filename + '.csv'
            elif format is not 'binary' and format is not 'json':
                raise ValueError("Invalid format: {}. Supported formats are 'csv' and 'binary' and 'json'".format(format))

        ## Save the SFrame
        url = _make_internal_url(filename)

        with cython_context():
            if format is 'binary':
                self.__proxy__.save(url)
            elif format is 'csv':
                assert filename.endswith(('.csv', '.csv.gz'))
                self.__proxy__.save_as_csv(url, {})
            elif format is 'json':
                self.export_json(url)
            else:
                raise ValueError("Unsupported format: {}".format(format))

    def export_csv(self, filename, delimiter=',', line_terminator='\n',
            header=True, quote_level=csv.QUOTE_NONNUMERIC, double_quote=True,
            escape_char='\\', quote_char='\"', na_rep='',
            file_header='', file_footer='', line_prefix='',
            _no_prefix_on_first_value=False, **kwargs):
        """
        Writes an SFrame to a CSV file.

        Parameters
        ----------
        filename : string
            The location to save the CSV.

        delimiter : string, optional
            This describes the delimiter used for writing csv files.

        line_terminator: string, optional
            The newline character

        header : bool, optional
            If true, the column names are emitted as a header.

        quote_level: csv.QUOTE_ALL | csv.QUOTE_NONE | csv.QUOTE_NONNUMERIC, optional
            The quoting level. If csv.QUOTE_ALL, every field is quoted.
            if csv.quote_NONE, no field is quoted. If csv.QUOTE_NONNUMERIC, only
            non-numeric fileds are quoted. csv.QUOTE_MINIMAL is interpreted as
            csv.QUOTE_NONNUMERIC.

        double_quote : bool, optional
            If True, quotes are escaped as two consecutive quotes

        escape_char : string, optional
            Character which begins a C escape sequence

        quote_char: string, optional
            Character used to quote fields

        na_rep: string, optional
            The value used to denote a missing value.

        file_header: string, optional
            A string printed to the start of the file

        file_footer: string, optional
            A string printed to the end of the file

        line_prefix: string, optional
            A string printed at the start of each value line
        """
        # Pandas argument compatibility
        if "sep" in kwargs:
            delimiter = kwargs['sep']
            del kwargs['sep']
        if "quotechar" in kwargs:
            quote_char = kwargs['quotechar']
            del kwargs['quotechar']
        if "doublequote" in kwargs:
            double_quote = kwargs['doublequote']
            del kwargs['doublequote']
        if "lineterminator" in kwargs:
            line_terminator = kwargs['lineterminator']
            del kwargs['lineterminator']
        if len(kwargs) > 0:
            raise TypeError("Unexpected keyword arguments " + str(list(kwargs.keys())))

        write_csv_options = {}
        write_csv_options['delimiter'] = delimiter
        write_csv_options['escape_char'] = escape_char
        write_csv_options['double_quote'] = double_quote
        write_csv_options['quote_char'] = quote_char
        if quote_level == csv.QUOTE_MINIMAL:
            write_csv_options['quote_level'] = 0
        elif quote_level == csv.QUOTE_ALL:
            write_csv_options['quote_level'] = 1
        elif quote_level == csv.QUOTE_NONNUMERIC:
            write_csv_options['quote_level'] = 2
        elif quote_level == csv.QUOTE_NONE:
            write_csv_options['quote_level'] = 3
        write_csv_options['header'] = header
        write_csv_options['line_terminator'] = line_terminator
        write_csv_options['na_value'] = na_rep
        write_csv_options['file_header'] = file_header
        write_csv_options['file_footer'] = file_footer
        write_csv_options['line_prefix'] = line_prefix

        # undocumented option. Disables line prefix on the first value line
        write_csv_options['_no_prefix_on_first_value'] = _no_prefix_on_first_value

        url = _make_internal_url(filename)
        self.__proxy__.save_as_csv(url, write_csv_options)

    def export_json(self,
                    filename,
                    orient='records'):
        """
        Writes an SFrame to a JSON file.

        Parameters
        ----------
        filename : string
            The location to save the JSON file.

        orient : string, optional. Either "records" or "lines"
            If orient="records" the file is saved as a single JSON array.
            If orient="lines", the file is saves as a JSON value per line.

        Examples
        --------
        The orient parameter describes the expected input format of the JSON
        file.

        If orient="records", the output will be a single JSON Array where
        each array element is a dictionary describing the row.

        >>> g
        Columns:
                a	int
                b	int
        Rows: 3
        Data:
        +---+---+
        | a | b |
        +---+---+
        | 1 | 1 |
        | 2 | 2 |
        | 3 | 3 |
        +---+---+
        >>> g.export('output.json', orient='records')
        >>> !cat output.json
        [
        {'a':1,'b':1},
        {'a':2,'b':2},
        {'a':3,'b':3},
        ]

        If orient="rows", each row will be emitted as a JSON dictionary to
        each file line.

        >>> g
        Columns:
                a	int
                b	int
        Rows: 3
        Data:
        +---+---+
        | a | b |
        +---+---+
        | 1 | 1 |
        | 2 | 2 |
        | 3 | 3 |
        +---+---+
        >>> g.export('output.json', orient='rows')
        >>> !cat output.json
        {'a':1,'b':1}
        {'a':2,'b':2}
        {'a':3,'b':3}
        """
        if orient == "records":
            self.pack_columns(dtype=dict).export_csv(
                    filename, file_header='[', file_footer=']',
                    header=False, double_quote=False,
                    quote_level=csv.QUOTE_NONE,
                    line_prefix=',',
                    _no_prefix_on_first_value=True)
        elif orient == "lines":
            self.pack_columns(dtype=dict).export_csv(
                    filename, header=False, double_quote=False, quote_level=csv.QUOTE_NONE)
        else:
            raise ValueError("Invalid value for orient parameter (" + str(orient) + ")")

    def _save_reference(self, filename):
        """
        Performs an incomplete save of an existing SFrame into a directory.
        This saved SFrame may reference SFrames in other locations in the same
        filesystem for certain resources.

        Parameters
        ----------
        filename : string
            The location to save the SFrame. Either a local directory or a
            remote URL.

        See Also
        --------
        load_sframe, SFrame

        Examples
        --------
        >>> # Save the sframe into binary format
        >>> sf.save_reference('data/training_data_sframe')
        """
        ## Save the SFrame
        url = _make_internal_url(filename)

        with cython_context():
            self.__proxy__.save_reference(url)


    def select_column(self, column_name):
        """
        Get a reference to the :class:`~turicreate.SArray` that corresponds with
        the given column_name. Throws an exception if the column_name is
        something other than a string or if the column name is not found.

        Parameters
        ----------
        column_name: str
            The column name.

        Returns
        -------
        out : SArray
            The SArray that is referred by ``column_name``.

        See Also
        --------
        select_columns

        Examples
        --------
        >>> sf = turicreate.SFrame({'user_id': [1,2,3],
        ...                       'user_name': ['alice', 'bob', 'charlie']})
        >>> # This line is equivalent to `sa = sf['user_name']`
        >>> sa = sf.select_column('user_name')
        >>> sa
        dtype: str
        Rows: 3
        ['alice', 'bob', 'charlie']
        """
        if not isinstance(column_name, str):
            raise TypeError("Invalid column_nametype: must be str")
        with cython_context():
            return SArray(data=[], _proxy=self.__proxy__.select_column(column_name))

    def select_columns(self, column_names):
        """
        Selects all columns where the name of the column or the type of column
        is included in the column_names. An exception is raised if duplicate columns
        are selected i.e. sf.select_columns(['a','a']), or non-existent columns
        are selected.

        Throws an exception for all other input types.

        Parameters
        ----------
        column_names: list[str or type]
            The list of column names or a list of types.

        Returns
        -------
        out : SFrame
            A new SFrame that is made up of the columns referred to in
            ``column_names`` from the current SFrame.

        See Also
        --------
        select_column

        Examples
        --------
        >>> sf = turicreate.SFrame({'user_id': [1,2,3],
        ...                       'user_name': ['alice', 'bob', 'charlie'],
        ...                       'zipcode': [98101, 98102, 98103]
        ...                      })
        >>> # This line is equivalent to `sf2 = sf[['user_id', 'zipcode']]`
        >>> sf2 = sf.select_columns(['user_id', 'zipcode'])
        >>> sf2
        +---------+---------+
        | user_id | zipcode |
        +---------+---------+
        |    1    |  98101  |
        |    2    |  98102  |
        |    3    |  98103  |
        +---------+---------+
        [3 rows x 2 columns]
        """
        if not _is_non_string_iterable(column_names):
            raise TypeError("column_names must be an iterable")
        if not (all([isinstance(x, six.string_types) or isinstance(x, type) or isinstance(x, bytes)
                     for x in column_names])):
            raise TypeError("Invalid key type: must be str, unicode, bytes or type")

        requested_str_columns = [s for s in column_names if isinstance(s, six.string_types)]

        # Make sure there are no duplicates keys
        from collections import Counter
        column_names_counter = Counter(column_names)
        if (len(column_names)) != len(column_names_counter):
            for key in column_names_counter:
                if column_names_counter[key] > 1:
                    raise ValueError("There are duplicate keys in key list: '" + key + "'")

        colnames_and_types = list(zip(self.column_names(), self.column_types()))

        # Ok. we want the string columns to be in the ordering defined by the
        # argument.  And then all the type selection columns.
        selected_columns = requested_str_columns
        typelist = [s for s in column_names if isinstance(s, type)]

        # next the type selection columns
        # loop through all the columns, adding all columns with types in
        # typelist. But don't add a column if it has already been added.
        for i in colnames_and_types:
            if i[1] in typelist and i[0] not in selected_columns:
                selected_columns += [i[0]]

        selected_columns = selected_columns

        with cython_context():
            return SFrame(data=[], _proxy=self.__proxy__.select_columns(selected_columns))

    def add_column(self, data, column_name="", inplace=False):
        """
        Returns an SFrame with a new column. The number of elements in the data
        given must match the length of every other column of the SFrame.
        If no name is given, a default name is chosen.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        data : SArray
            The 'column' of data to add.

        column_name : string, optional
            The name of the column. If no name is given, a default name is
            chosen.

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The current SFrame.

        See Also
        --------
        add_columns

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})
        >>> sa = turicreate.SArray(['cat', 'dog', 'fossa'])
        >>> # This line is equivalent to `sf['species'] = sa`
        >>> res = sf.add_column(sa, 'species')
        >>> res
        +----+-----+---------+
        | id | val | species |
        +----+-----+---------+
        | 1  |  A  |   cat   |
        | 2  |  B  |   dog   |
        | 3  |  C  |  fossa  |
        +----+-----+---------+
        [3 rows x 3 columns]
        """
        # Check type for pandas dataframe or SArray?
        if not isinstance(data, SArray):
            raise TypeError("Must give column as SArray")
        if not isinstance(column_name, str):
            raise TypeError("Invalid column name: must be str")

        if inplace:
            ret = self
        else:
            ret = self.copy()

        with cython_context():
            ret.__proxy__.add_column(data.__proxy__, column_name)

        ret._cache = None
        return ret

    def add_columns(self, data, column_names=None, inplace=False):
        """
        Returns an SFrame with multiple columns added. The number of
        elements in all columns must match the length of every other column of
        the SFrame.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        data : list[SArray] or SFrame
            The columns to add.

        column_names: list of string, optional
            A list of column names. All names must be specified. ``column_names`` is
            ignored if data is an SFrame.

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The current SFrame.

        See Also
        --------
        add_column

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})
        >>> sf2 = turicreate.SFrame({'species': ['cat', 'dog', 'fossa'],
        ...                        'age': [3, 5, 9]})
        >>> res = sf.add_columns(sf2)
        >>> res
        +----+-----+-----+---------+
        | id | val | age | species |
        +----+-----+-----+---------+
        | 1  |  A  |  3  |   cat   |
        | 2  |  B  |  5  |   dog   |
        | 3  |  C  |  9  |  fossa  |
        +----+-----+-----+---------+
        [3 rows x 4 columns]
        """
        datalist = data
        if isinstance(data, SFrame):
            other = data
            datalist = [other.select_column(name) for name in other.column_names()]
            column_names = other.column_names()

            my_columns = set(self.column_names())
            for name in column_names:
                if name in my_columns:
                    raise ValueError("Column '" + name + "' already exists in current SFrame")
        else:
            if not _is_non_string_iterable(datalist):
                raise TypeError("datalist must be an iterable")
            if not _is_non_string_iterable(column_names):
                raise TypeError("column_names must be an iterable")

            if not all([isinstance(x, SArray) for x in datalist]):
                raise TypeError("Must give column as SArray")
            if not all([isinstance(x, str) for x in column_names]):
                raise TypeError("Invalid column name in list : must all be str")

        if inplace:
            ret = self
        else:
            ret = self.copy()

        with cython_context():
            ret.__proxy__.add_columns([x.__proxy__ for x in datalist], column_names)

        ret._cache = None
        return ret

    def remove_column(self, column_name, inplace=False):
        """
        Returns an SFrame with a column removed.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        column_name : string
            The name of the column to remove.

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The SFrame with given column removed.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})
        >>> # This is equivalent to `del sf['val']`
        >>> res = sf.remove_column('val')
        >>> res
        +----+
        | id |
        +----+
        | 1  |
        | 2  |
        | 3  |
        +----+
        [3 rows x 1 columns]
        """
        column_name = str(column_name)
        if column_name not in self.column_names():
            raise KeyError('Cannot find column %s' % column_name)
        colid = self.column_names().index(column_name)

        if inplace:
            ret = self
        else:
            ret = self.copy()

        with cython_context():
            ret.__proxy__.remove_column(colid)

        ret._cache = None
        return ret

    def remove_columns(self, column_names, inplace=False):
        """
        Returns an SFrame with one or more columns removed.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        column_names : list or iterable
            A list or iterable of column names.

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The SFrame with given columns removed.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val1': ['A', 'B', 'C'], 'val2' : [10, 11, 12]})
        >>> res = sf.remove_columns(['val1', 'val2'])
        >>> res
        +----+
        | id |
        +----+
        | 1  |
        | 2  |
        | 3  |
        +----+
        [3 rows x 1 columns]
        """
        column_names = list(column_names)
        existing_columns = dict((k, i) for i, k in enumerate(self.column_names()))

        for name in column_names:
            if name not in existing_columns:
                raise KeyError('Cannot find column %s' % name)

        # Delete it going backwards so we don't invalidate indices
        deletion_indices = sorted(existing_columns[name] for name in column_names)

        if inplace:
            ret = self
        else:
            ret = self.copy()

        for colid in reversed(deletion_indices):
            with cython_context():
                ret.__proxy__.remove_column(colid)

        ret._cache = None
        return ret


    def swap_columns(self, column_name_1, column_name_2, inplace=False):
        """
        Returns an SFrame with two column positions swapped.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        column_name_1 : string
            Name of column to swap

        column_name_2 : string
            Name of other column to swap

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The SFrame with swapped columns.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})
        >>> res = sf.swap_columns('id', 'val')
        >>> res
        +-----+-----+
        | val | id  |
        +-----+-----+
        |  A  |  1  |
        |  B  |  2  |
        |  C  |  3  |
        +----+-----+
        [3 rows x 2 columns]
        """
        colnames = self.column_names()
        colid_1 = colnames.index(column_name_1)
        colid_2 = colnames.index(column_name_2)

        if inplace:
            ret = self
        else:
            ret = self.copy()

        with cython_context():
            ret.__proxy__.swap_columns(colid_1, colid_2)

        ret._cache = None
        return ret

    def rename(self, names, inplace=False):
        """
        Returns an SFrame with columns renamed. ``names`` is expected to be a
        dict specifying the old and new names. This changes the names of the
        columns given as the keys and replaces them with the names given as the
        values.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        names : dict [string, string]
            Dictionary of [old_name, new_name]

        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The current SFrame.

        See Also
        --------
        column_names

        Examples
        --------
        >>> sf = SFrame({'X1': ['Alice','Bob'],
        ...              'X2': ['123 Fake Street','456 Fake Street']})
        >>> res = sf.rename({'X1': 'name', 'X2':'address'})
        >>> res
        +-------+-----------------+
        |  name |     address     |
        +-------+-----------------+
        | Alice | 123 Fake Street |
        |  Bob  | 456 Fake Street |
        +-------+-----------------+
        [2 rows x 2 columns]
        """
        if (type(names) is not dict):
            raise TypeError('names must be a dictionary: oldname -> newname')
        all_columns = set(self.column_names())
        for k in names:
            if not k in all_columns:
                raise ValueError('Cannot find column %s in the SFrame' % k)

        if inplace:
            ret = self
        else:
            ret = self.copy()

        with cython_context():
            for k in names:
                colid = ret.column_names().index(k)
                ret.__proxy__.set_column_name(colid, names[k])
        ret._cache = None
        return ret

    def __getitem__(self, key):
        """
        This method does things based on the type of `key`.

        If `key` is:
            * str
                selects column with name 'key'
            * type
                selects all columns with types matching the type
            * list of str or type
                selects all columns with names or type in the list
            * SArray
                Performs a logical filter.  Expects given SArray to be the same
                length as all columns in current SFrame.  Every row
                corresponding with an entry in the given SArray that is
                equivalent to False is filtered from the result.
            * int
                Returns a single row of the SFrame (the `key`th one) as a dictionary.
            * slice
                Returns an SFrame including only the sliced rows.
        """
        if type(key) is SArray:
            return self._row_selector(key)
        elif isinstance(key, six.string_types):
            if six.PY2 and type(key) == unicode:
                key = key.encode('utf-8')
            return self.select_column(key)
        elif type(key) is type:
            return self.select_columns([key])
        elif _is_non_string_iterable(key):
            return self.select_columns(key)
        elif isinstance(key, numbers.Integral):
            sf_len = len(self)

            if key < 0:
                key = sf_len + key
            if key >= sf_len:
                raise IndexError("SFrame index out of range")

            if not hasattr(self, '_cache') or self._cache is None:
                self._cache = {}

            try:
                lb, ub, value_list = self._cache["getitem_cache"]
                if lb <= key < ub:
                    return value_list[int(key - lb)]

            except KeyError:
                pass

            # Not in cache, need to grab it.  Smaller here than with sarray

            # Do we have a good block size that won't cause memory to blow up?
            if not "getitem_cache_blocksize" in self._cache:
                block_size = \
                  (8*1024) // sum( (2 if dt in [int, long, float] else 8) for dt in self.column_types())

                block_size = max(16, block_size)
                self._cache["getitem_cache_blocksize"] = block_size
            else:
                block_size = self._cache["getitem_cache_blocksize"]

            block_num = int(key // block_size)

            lb = block_num * block_size
            ub = min(sf_len, lb + block_size)

            val_list = list(SFrame(_proxy = self.__proxy__.copy_range(lb, 1, ub)))
            self._cache["getitem_cache"] = (lb, ub, val_list)
            return val_list[int(key - lb)]

        elif type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            if step is None:
                step = 1
            # handle negative indices
            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop
            return SFrame(_proxy = self.__proxy__.copy_range(start, step, stop))
        else:
            raise TypeError("Invalid index type: must be SArray, list, int, or str")

    def __setitem__(self, key, value):
        """
        A wrapper around add_column(s).  Key can be either a list or a str.  If
        value is an SArray, it is added to the SFrame as a column.  If it is a
        constant value (int, str, or float), then a column is created where
        every entry is equal to the constant value.  Existing columns can also
        be replaced using this wrapper.
        """
        if type(key) is list:
            self.add_columns(value, key, inplace=True)
        elif type(key) is str:
            sa_value = None
            if (type(value) is SArray):
                sa_value = value
            elif _is_non_string_iterable(value):  # wrap list, array... to sarray
                sa_value = SArray(value)
            else:  # create an sarray  of constant value
                sa_value = SArray.from_const(value, self.num_rows())

            # set new column
            if not key in self.column_names():
                with cython_context():
                    self.add_column(sa_value, key, inplace=True)
            else:
                # special case if replacing the only column.
                # server would fail the replacement if the new column has different
                # length than current one, which doesn't make sense if we are replacing
                # the only column. To support this, we first take out the only column
                # and then put it back if exception happens
                single_column = (self.num_columns() == 1)
                if (single_column):
                    tmpname = key
                    saved_column = self.select_column(key)
                    self.remove_column(key, inplace=True)
                else:
                    # add the column to a unique column name.
                    tmpname = '__' + '-'.join(self.column_names())
                try:
                    self.add_column(sa_value, tmpname, inplace=True)
                except Exception:
                    if (single_column):
                        self.add_column(saved_column, key, inplace=True)
                    raise

                if (not single_column):
                    # if add succeeded, remove the column name and rename tmpname->columnname.
                    self.swap_columns(key, tmpname, inplace=True)
                    self.remove_column(key, inplace=True)
                    self.rename({tmpname: key}, inplace=True)

        else:
            raise TypeError('Cannot set column with key type ' + str(type(key)))

    def __delitem__(self, key):
        """
        Wrapper around remove_column.
        """
        self.remove_column(key, inplace=True)

    def materialize(self):
        """
        For an SFrame that is lazily evaluated, force the persistence of the
        SFrame to disk, committing all lazy evaluated operations.
        """
        return self.__materialize__()

    def __materialize__(self):
        """
        For an SFrame that is lazily evaluated, force the persistence of the
        SFrame to disk, committing all lazy evaluated operations.
        """
        with cython_context():
            self.__proxy__.materialize()

    def is_materialized(self):
        """
        Returns whether or not the SFrame has been materialized.
        """
        return self.__is_materialized__()

    def __is_materialized__(self):
        """
        Returns whether or not the SFrame has been materialized.
        """
        return self.__proxy__.is_materialized()

    def __has_size__(self):
        """
        Returns whether or not the size of the SFrame is known.
        """
        return self.__proxy__.has_size()

    def __query_plan_str__(self):
        """
        Returns the query plan as a dot graph string
        """
        return self.__proxy__.query_plan_string()

    def __iter__(self):
        """
        Provides an iterator to the rows of the SFrame.
        """


        def generator():
            elems_at_a_time = 262144
            self.__proxy__.begin_iterator()
            ret = self.__proxy__.iterator_get_next(elems_at_a_time)
            column_names = self.column_names()
            while(True):
                for j in ret:
                    yield dict(list(zip(column_names, j)))

                if len(ret) == elems_at_a_time:
                    ret = self.__proxy__.iterator_get_next(elems_at_a_time)
                else:
                    break

        return generator()

    def append(self, other):
        """
        Add the rows of an SFrame to the end of this SFrame.

        Both SFrames must have the same set of columns with the same column
        names and column types.

        Parameters
        ----------
        other : SFrame
            Another SFrame whose rows are appended to the current SFrame.

        Returns
        -------
        out : SFrame
            The result SFrame from the append operation.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [4, 6, 8], 'val': ['D', 'F', 'H']})
        >>> sf2 = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})
        >>> sf = sf.append(sf2)
        >>> sf
        +----+-----+
        | id | val |
        +----+-----+
        | 4  |  D  |
        | 6  |  F  |
        | 8  |  H  |
        | 1  |  A  |
        | 2  |  B  |
        | 3  |  C  |
        +----+-----+
        [6 rows x 2 columns]
        """
        if type(other) is not SFrame:
            raise RuntimeError("SFrame append can only work with SFrame")

        left_empty = len(self.column_names()) == 0
        right_empty = len(other.column_names()) == 0

        if (left_empty and right_empty):
            return SFrame()

        if (left_empty or right_empty):
            non_empty_sframe = self if right_empty else other
            return non_empty_sframe.__copy__()

        my_column_names = self.column_names()
        my_column_types = self.column_types()
        other_column_names = other.column_names()
        if (len(my_column_names) != len(other_column_names)):
            raise RuntimeError("Two SFrames have to have the same number of columns")

        # check if the order of column name is the same
        column_name_order_match = True
        for i in range(len(my_column_names)):
            if other_column_names[i] != my_column_names[i]:
                column_name_order_match = False
                break

        processed_other_frame = other
        if not column_name_order_match:
            # we allow name order of two sframes to be different, so we create a new sframe from
            # "other" sframe to make it has exactly the same shape
            processed_other_frame = SFrame()
            for i in range(len(my_column_names)):
                col_name = my_column_names[i]
                if(col_name not in other_column_names):
                    raise RuntimeError("Column " + my_column_names[i] + " does not exist in second SFrame")

                other_column = other.select_column(col_name)
                processed_other_frame.add_column(other_column, col_name, inplace=True)

                # check column type
                if my_column_types[i] != other_column.dtype:
                    raise RuntimeError("Column " + my_column_names[i] + " type is not the same in two SFrames, one is " + str(my_column_types[i]) + ", the other is " + str(other_column.dtype))

        with cython_context():
            return SFrame(_proxy=self.__proxy__.append(processed_other_frame.__proxy__))

    def groupby(self, key_column_names, operations, *args):
        """
        Perform a group on the key_column_names followed by aggregations on the
        columns listed in operations.

        The operations parameter is a dictionary that indicates which
        aggregation operators to use and which columns to use them on. The
        available operators are SUM, MAX, MIN, COUNT, AVG, VAR, STDV, CONCAT,
        SELECT_ONE, ARGMIN, ARGMAX, and QUANTILE. For convenience, aggregators
        MEAN, STD, and VARIANCE are available as synonyms for AVG, STDV, and
        VAR. See :mod:`~turicreate.aggregate` for more detail on the aggregators.

        Parameters
        ----------
        key_column_names : string | list[string]
            Column(s) to group by. Key columns can be of any type other than
            dictionary.

        operations : dict, list
            Dictionary of columns and aggregation operations. Each key is a
            output column name and each value is an aggregator. This can also
            be a list of aggregators, in which case column names will be
            automatically assigned.

        *args
            All other remaining arguments will be interpreted in the same
            way as the operations argument.

        Returns
        -------
        out_sf : SFrame
            A new SFrame, with a column for each groupby column and each
            aggregation operation.

        See Also
        --------
        aggregate

        Notes
        -----
        * Numeric aggregators (such as sum, mean, stdev etc.) follow the skip
        None policy i.e they will omit all missing values from the aggregation.
        As an example, `sum([None, 5, 10]) = 15` because the `None` value is
        skipped.
        * Aggregators have a default value when no values (after skipping all
        `None` values) are present. Default values are `None` for ['ARGMAX',
        'ARGMIN', 'AVG', 'STD', 'MEAN', 'MIN', 'MAX'],  `0` for ['COUNT'
        'COUNT_DISTINCT', 'DISTINCT'] `[]` for 'CONCAT', 'QUANTILE',
        'DISTINCT', and `{}` for 'FREQ_COUNT'.

        Examples
        --------
        Suppose we have an SFrame with movie ratings by many users.

        >>> import turicreate.aggregate as agg
        >>> url = 'https://static.turi.com/datasets/rating_data_example.csv'
        >>> sf = turicreate.SFrame.read_csv(url)
        >>> sf
        +---------+----------+--------+
        | user_id | movie_id | rating |
        +---------+----------+--------+
        |  25904  |   1663   |   3    |
        |  25907  |   1663   |   3    |
        |  25923  |   1663   |   3    |
        |  25924  |   1663   |   3    |
        |  25928  |   1663   |   2    |
        |  25933  |   1663   |   4    |
        |  25934  |   1663   |   4    |
        |  25935  |   1663   |   4    |
        |  25936  |   1663   |   5    |
        |  25937  |   1663   |   2    |
        |   ...   |   ...    |  ...   |
        +---------+----------+--------+
        [10000 rows x 3 columns]

        Compute the number of occurrences of each user.

        >>> user_count = sf.groupby(key_column_names='user_id',
        ...                         operations={'count': agg.COUNT()})
        >>> user_count
        +---------+-------+
        | user_id | count |
        +---------+-------+
        |  62361  |   1   |
        |  30727  |   1   |
        |  40111  |   1   |
        |  50513  |   1   |
        |  35140  |   1   |
        |  42352  |   1   |
        |  29667  |   1   |
        |  46242  |   1   |
        |  58310  |   1   |
        |  64614  |   1   |
        |   ...   |  ...  |
        +---------+-------+
        [9852 rows x 2 columns]

        Compute the mean and standard deviation of ratings per user.

        >>> user_rating_stats = sf.groupby(key_column_names='user_id',
        ...                                operations={
        ...                                    'mean_rating': agg.MEAN('rating'),
        ...                                    'std_rating': agg.STD('rating')
        ...                                })
        >>> user_rating_stats
        +---------+-------------+------------+
        | user_id | mean_rating | std_rating |
        +---------+-------------+------------+
        |  62361  |     5.0     |    0.0     |
        |  30727  |     4.0     |    0.0     |
        |  40111  |     2.0     |    0.0     |
        |  50513  |     4.0     |    0.0     |
        |  35140  |     4.0     |    0.0     |
        |  42352  |     5.0     |    0.0     |
        |  29667  |     4.0     |    0.0     |
        |  46242  |     5.0     |    0.0     |
        |  58310  |     2.0     |    0.0     |
        |  64614  |     2.0     |    0.0     |
        |   ...   |     ...     |    ...     |
        +---------+-------------+------------+
        [9852 rows x 3 columns]

        Compute the movie with the minimum rating per user.

        >>> chosen_movies = sf.groupby(key_column_names='user_id',
        ...                            operations={
        ...                                'worst_movies': agg.ARGMIN('rating','movie_id')
        ...                            })
        >>> chosen_movies
        +---------+-------------+
        | user_id | worst_movies |
        +---------+-------------+
        |  62361  |     1663    |
        |  30727  |     1663    |
        |  40111  |     1663    |
        |  50513  |     1663    |
        |  35140  |     1663    |
        |  42352  |     1663    |
        |  29667  |     1663    |
        |  46242  |     1663    |
        |  58310  |     1663    |
        |  64614  |     1663    |
        |   ...   |     ...     |
        +---------+-------------+
        [9852 rows x 2 columns]

        Compute the movie with the max rating per user and also the movie with
        the maximum imdb-ranking per user.

        >>> sf['imdb-ranking'] = sf['rating'] * 10
        >>> chosen_movies = sf.groupby(key_column_names='user_id',
        ...         operations={('max_rating_movie','max_imdb_ranking_movie'): agg.ARGMAX(('rating','imdb-ranking'),'movie_id')})
        >>> chosen_movies
        +---------+------------------+------------------------+
        | user_id | max_rating_movie | max_imdb_ranking_movie |
        +---------+------------------+------------------------+
        |  62361  |       1663       |          16630         |
        |  30727  |       1663       |          16630         |
        |  40111  |       1663       |          16630         |
        |  50513  |       1663       |          16630         |
        |  35140  |       1663       |          16630         |
        |  42352  |       1663       |          16630         |
        |  29667  |       1663       |          16630         |
        |  46242  |       1663       |          16630         |
        |  58310  |       1663       |          16630         |
        |  64614  |       1663       |          16630         |
        |   ...   |       ...        |          ...           |
        +---------+------------------+------------------------+
        [9852 rows x 3 columns]

        Compute the movie with the max rating per user.

        >>> chosen_movies = sf.groupby(key_column_names='user_id',
                    operations={'best_movies': agg.ARGMAX('rating','movie')})

        Compute the movie with the max rating per user and also the movie with the maximum imdb-ranking per user.

        >>> chosen_movies = sf.groupby(key_column_names='user_id',
                   operations={('max_rating_movie','max_imdb_ranking_movie'): agg.ARGMAX(('rating','imdb-ranking'),'movie')})

        Compute the count, mean, and standard deviation of ratings per (user,
        time), automatically assigning output column names.

        >>> sf['time'] = sf.apply(lambda x: (x['user_id'] + x['movie_id']) % 11 + 2000)
        >>> user_rating_stats = sf.groupby(['user_id', 'time'],
        ...                                [agg.COUNT(),
        ...                                 agg.AVG('rating'),
        ...                                 agg.STDV('rating')])
        >>> user_rating_stats
        +------+---------+-------+---------------+----------------+
        | time | user_id | Count | Avg of rating | Stdv of rating |
        +------+---------+-------+---------------+----------------+
        | 2006 |  61285  |   1   |      4.0      |      0.0       |
        | 2000 |  36078  |   1   |      4.0      |      0.0       |
        | 2003 |  47158  |   1   |      3.0      |      0.0       |
        | 2007 |  34446  |   1   |      3.0      |      0.0       |
        | 2010 |  47990  |   1   |      3.0      |      0.0       |
        | 2003 |  42120  |   1   |      5.0      |      0.0       |
        | 2007 |  44940  |   1   |      4.0      |      0.0       |
        | 2008 |  58240  |   1   |      4.0      |      0.0       |
        | 2002 |   102   |   1   |      1.0      |      0.0       |
        | 2009 |  52708  |   1   |      3.0      |      0.0       |
        | ...  |   ...   |  ...  |      ...      |      ...       |
        +------+---------+-------+---------------+----------------+
        [10000 rows x 5 columns]


        The groupby function can take a variable length list of aggregation
        specifiers so if we want the count and the 0.25 and 0.75 quantiles of
        ratings:

        >>> user_rating_stats = sf.groupby(['user_id', 'time'], agg.COUNT(),
        ...                                {'rating_quantiles': agg.QUANTILE('rating',[0.25, 0.75])})
        >>> user_rating_stats
        +------+---------+-------+------------------------+
        | time | user_id | Count |    rating_quantiles    |
        +------+---------+-------+------------------------+
        | 2006 |  61285  |   1   | array('d', [4.0, 4.0]) |
        | 2000 |  36078  |   1   | array('d', [4.0, 4.0]) |
        | 2003 |  47158  |   1   | array('d', [3.0, 3.0]) |
        | 2007 |  34446  |   1   | array('d', [3.0, 3.0]) |
        | 2010 |  47990  |   1   | array('d', [3.0, 3.0]) |
        | 2003 |  42120  |   1   | array('d', [5.0, 5.0]) |
        | 2007 |  44940  |   1   | array('d', [4.0, 4.0]) |
        | 2008 |  58240  |   1   | array('d', [4.0, 4.0]) |
        | 2002 |   102   |   1   | array('d', [1.0, 1.0]) |
        | 2009 |  52708  |   1   | array('d', [3.0, 3.0]) |
        | ...  |   ...   |  ...  |          ...           |
        +------+---------+-------+------------------------+
        [10000 rows x 4 columns]

        To put all items a user rated into one list value by their star rating:

        >>> user_rating_stats = sf.groupby(["user_id", "rating"],
        ...                                {"rated_movie_ids":agg.CONCAT("movie_id")})
        >>> user_rating_stats
        +--------+---------+----------------------+
        | rating | user_id |     rated_movie_ids  |
        +--------+---------+----------------------+
        |   3    |  31434  | array('d', [1663.0]) |
        |   5    |  25944  | array('d', [1663.0]) |
        |   4    |  38827  | array('d', [1663.0]) |
        |   4    |  51437  | array('d', [1663.0]) |
        |   4    |  42549  | array('d', [1663.0]) |
        |   4    |  49532  | array('d', [1663.0]) |
        |   3    |  26124  | array('d', [1663.0]) |
        |   4    |  46336  | array('d', [1663.0]) |
        |   4    |  52133  | array('d', [1663.0]) |
        |   5    |  62361  | array('d', [1663.0]) |
        |  ...   |   ...   |         ...          |
        +--------+---------+----------------------+
        [9952 rows x 3 columns]

        To put all items and rating of a given user together into a dictionary
        value:

        >>> user_rating_stats = sf.groupby("user_id",
        ...                                {"movie_rating":agg.CONCAT("movie_id", "rating")})
        >>> user_rating_stats
        +---------+--------------+
        | user_id | movie_rating |
        +---------+--------------+
        |  62361  |  {1663: 5}   |
        |  30727  |  {1663: 4}   |
        |  40111  |  {1663: 2}   |
        |  50513  |  {1663: 4}   |
        |  35140  |  {1663: 4}   |
        |  42352  |  {1663: 5}   |
        |  29667  |  {1663: 4}   |
        |  46242  |  {1663: 5}   |
        |  58310  |  {1663: 2}   |
        |  64614  |  {1663: 2}   |
        |   ...   |     ...      |
        +---------+--------------+
        [9852 rows x 2 columns]
        """
        # some basic checking first
        # make sure key_column_names is a list
        if isinstance(key_column_names, str):
            key_column_names = [key_column_names]
        # check that every column is a string, and is a valid column name
        my_column_names = self.column_names()
        key_columns_array = []
        for column in key_column_names:
            if not isinstance(column, str):
                raise TypeError("Column name must be a string")
            if column not in my_column_names:
                raise KeyError("Column " + column + " does not exist in SFrame")
            if self[column].dtype == dict:
                raise TypeError("Cannot group on a dictionary column.")
            key_columns_array.append(column)

        group_output_columns = []
        group_columns = []
        group_ops = []

        all_ops = [operations] + list(args)
        for op_entry in all_ops:
            # if it is not a dict, nor a list, it is just a single aggregator
            # element (probably COUNT). wrap it in a list so we can reuse the
            # list processing code
            operation = op_entry
            if not(isinstance(operation, list) or isinstance(operation, dict)):
              operation = [operation]
            if isinstance(operation, dict):
              # now sweep the dict and add to group_columns and group_ops
              for key in operation:
                  val = operation[key]
                  if type(val) is tuple:
                    (op, column) = val
                    if (op == '__builtin__avg__' and self[column[0]].dtype in [array.array, numpy.ndarray]):
                        op = '__builtin__vector__avg__'

                    if (op == '__builtin__sum__' and self[column[0]].dtype in [array.array, numpy.ndarray]):
                        op = '__builtin__vector__sum__'

                    if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and ((type(column[0]) is tuple) != (type(key) is tuple)):
                        raise TypeError("Output column(s) and aggregate column(s) for aggregate operation should be either all tuple or all string.")

                    if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and type(column[0]) is tuple:
                      for (col,output) in zip(column[0],key):
                        group_columns = group_columns + [[col,column[1]]]
                        group_ops = group_ops + [op]
                        group_output_columns = group_output_columns + [output]
                    else:
                      group_columns = group_columns + [column]
                      group_ops = group_ops + [op]
                      group_output_columns = group_output_columns + [key]

                    if (op == '__builtin__concat__dict__'):
                        key_column = column[0]
                        key_column_type = self.select_column(key_column).dtype
                        if not key_column_type in (int, float, str):
                            raise TypeError('CONCAT key column must be int, float or str type')

                  elif val == aggregate.COUNT:
                    group_output_columns = group_output_columns + [key]
                    val = aggregate.COUNT()
                    (op, column) = val
                    group_columns = group_columns + [column]
                    group_ops = group_ops + [op]
                  else:
                    raise TypeError("Unexpected type in aggregator definition of output column: " + key)
            elif isinstance(operation, list):
              # we will be using automatically defined column names
              for val in operation:
                  if type(val) is tuple:
                    (op, column) = val
                    if (op == '__builtin__avg__' and self[column[0]].dtype in [array.array, numpy.ndarray]):
                        op = '__builtin__vector__avg__'

                    if (op == '__builtin__sum__' and self[column[0]].dtype in [array.array, numpy.ndarray]):
                        op = '__builtin__vector__sum__'

                    if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and type(column[0]) is tuple:
                      for col in column[0]:
                        group_columns = group_columns + [[col,column[1]]]
                        group_ops = group_ops + [op]
                        group_output_columns = group_output_columns + [""]
                    else:
                      group_columns = group_columns + [column]
                      group_ops = group_ops + [op]
                      group_output_columns = group_output_columns + [""]

                    if (op == '__builtin__concat__dict__'):
                        key_column = column[0]
                        key_column_type = self.select_column(key_column).dtype
                        if not key_column_type in (int, float, str):
                            raise TypeError('CONCAT key column must be int, float or str type')

                  elif val == aggregate.COUNT:
                    group_output_columns = group_output_columns + [""]
                    val = aggregate.COUNT()
                    (op, column) = val
                    group_columns = group_columns + [column]
                    group_ops = group_ops + [op]
                  else:
                    raise TypeError("Unexpected type in aggregator definition.")


        # let's validate group_columns and group_ops are valid
        for (cols, op) in zip(group_columns, group_ops):
            for col in cols:
                if not isinstance(col, str):
                    raise TypeError("Column name must be a string")

            if not isinstance(op, str):
                raise TypeError("Operation type not recognized.")

            if op is not aggregate.COUNT()[0]:
                for col in cols:
                    if col not in my_column_names:
                        raise KeyError("Column " + col + " does not exist in SFrame")


        with cython_context():
            return SFrame(_proxy=self.__proxy__.groupby_aggregate(key_columns_array,
                                                                  group_columns,
                                                                  group_output_columns,
                                                                  group_ops))

    def join(self, right, on=None, how='inner'):
        """
        Merge two SFrames. Merges the current (left) SFrame with the given
        (right) SFrame using a SQL-style equi-join operation by columns.

        Parameters
        ----------
        right : SFrame
            The SFrame to join.

        on : None | str | list | dict, optional
            The column name(s) representing the set of join keys.  Each row that
            has the same value in this set of columns will be merged together.

            * If 'None' is given, join will use all columns that have the same
              name as the set of join keys.

            * If a str is given, this is interpreted as a join using one column,
              where both SFrames have the same column name.

            * If a list is given, this is interpreted as a join using one or
              more column names, where each column name given exists in both
              SFrames.

            * If a dict is given, each dict key is taken as a column name in the
              left SFrame, and each dict value is taken as the column name in
              right SFrame that will be joined together. e.g.
              {'left_col_name':'right_col_name'}.

        how : {'left', 'right', 'outer', 'inner'}, optional
            The type of join to perform.  'inner' is default.

            * inner: Equivalent to a SQL inner join.  Result consists of the
              rows from the two frames whose join key values match exactly,
              merged together into one SFrame.

            * left: Equivalent to a SQL left outer join. Result is the union
              between the result of an inner join and the rest of the rows from
              the left SFrame, merged with missing values.

            * right: Equivalent to a SQL right outer join.  Result is the union
              between the result of an inner join and the rest of the rows from
              the right SFrame, merged with missing values.

            * outer: Equivalent to a SQL full outer join. Result is
              the union between the result of a left outer join and a right
              outer join.

        Returns
        -------
        out : SFrame

        Examples
        --------
        >>> animals = turicreate.SFrame({'id': [1, 2, 3, 4],
        ...                           'name': ['dog', 'cat', 'sheep', 'cow']})
        >>> sounds = turicreate.SFrame({'id': [1, 3, 4, 5],
        ...                          'sound': ['woof', 'baa', 'moo', 'oink']})
        >>> animals.join(sounds, how='inner')
        +----+-------+-------+
        | id |  name | sound |
        +----+-------+-------+
        | 1  |  dog  |  woof |
        | 3  | sheep |  baa  |
        | 4  |  cow  |  moo  |
        +----+-------+-------+
        [3 rows x 3 columns]

        >>> animals.join(sounds, on='id', how='left')
        +----+-------+-------+
        | id |  name | sound |
        +----+-------+-------+
        | 1  |  dog  |  woof |
        | 3  | sheep |  baa  |
        | 4  |  cow  |  moo  |
        | 2  |  cat  |  None |
        +----+-------+-------+
        [4 rows x 3 columns]

        >>> animals.join(sounds, on=['id'], how='right')
        +----+-------+-------+
        | id |  name | sound |
        +----+-------+-------+
        | 1  |  dog  |  woof |
        | 3  | sheep |  baa  |
        | 4  |  cow  |  moo  |
        | 5  |  None |  oink |
        +----+-------+-------+
        [4 rows x 3 columns]

        >>> animals.join(sounds, on={'id':'id'}, how='outer')
        +----+-------+-------+
        | id |  name | sound |
        +----+-------+-------+
        | 1  |  dog  |  woof |
        | 3  | sheep |  baa  |
        | 4  |  cow  |  moo  |
        | 5  |  None |  oink |
        | 2  |  cat  |  None |
        +----+-------+-------+
        [5 rows x 3 columns]
        """
        available_join_types = ['left','right','outer','inner']

        if not isinstance(right, SFrame):
            raise TypeError("Can only join two SFrames")

        if how not in available_join_types:
            raise ValueError("Invalid join type")

        if (self.num_columns() <= 0) or (right.num_columns() <= 0):
            raise ValueError("Cannot join an SFrame with no columns.")

        join_keys = dict()
        if on is None:
            left_names = self.column_names()
            right_names = right.column_names()
            common_columns = [name for name in left_names if name in right_names]
            for name in common_columns:
                join_keys[name] = name
        elif type(on) is str:
            join_keys[on] = on
        elif type(on) is list:
            for name in on:
                if type(name) is not str:
                    raise TypeError("Join keys must each be a str.")
                join_keys[name] = name
        elif type(on) is dict:
            join_keys = on
        else:
            raise TypeError("Must pass a str, list, or dict of join keys")

        with cython_context():
            return SFrame(_proxy=self.__proxy__.join(right.__proxy__, how, join_keys))

    def filter_by(self, values, column_name, exclude=False):
        """
        Filter an SFrame by values inside an iterable object. Result is an
        SFrame that only includes (or excludes) the rows that have a column
        with the given ``column_name`` which holds one of the values in the
        given ``values`` :class:`~turicreate.SArray`. If ``values`` is not an
        SArray, we attempt to convert it to one before filtering.

        Parameters
        ----------
        values : SArray | list | numpy.ndarray | pandas.Series | str
            The values to use to filter the SFrame.  The resulting SFrame will
            only include rows that have one of these values in the given
            column.

        column_name : str
            The column of the SFrame to match with the given `values`.

        exclude : bool
            If True, the result SFrame will contain all rows EXCEPT those that
            have one of ``values`` in ``column_name``.

        Returns
        -------
        out : SFrame
            The filtered SFrame.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id': [1, 2, 3, 4],
        ...                      'animal_type': ['dog', 'cat', 'cow', 'horse'],
        ...                      'name': ['bob', 'jim', 'jimbob', 'bobjim']})
        >>> household_pets = ['cat', 'hamster', 'dog', 'fish', 'bird', 'snake']
        >>> sf.filter_by(household_pets, 'animal_type')
        +-------------+----+------+
        | animal_type | id | name |
        +-------------+----+------+
        |     dog     | 1  | bob  |
        |     cat     | 2  | jim  |
        +-------------+----+------+
        [2 rows x 3 columns]
        >>> sf.filter_by(household_pets, 'animal_type', exclude=True)
        +-------------+----+--------+
        | animal_type | id |  name  |
        +-------------+----+--------+
        |    horse    | 4  | bobjim |
        |     cow     | 3  | jimbob |
        +-------------+----+--------+
        [2 rows x 3 columns]
        """
        if type(column_name) is not str:
            raise TypeError("Must pass a str as column_name")

        existing_columns = self.column_names()
        if column_name not in existing_columns:
            raise KeyError("Column '" + column_name + "' not in SFrame.")

        if type(values) is not SArray:
            # If we were given a single element, try to put in list and convert
            # to SArray
            if not _is_non_string_iterable(values):
                values = [values]
            values = SArray(values)

        value_sf = SFrame()
        value_sf.add_column(values, column_name, inplace=True)

        existing_type = self.column_types()[self.column_names().index(column_name)]
        given_type = value_sf.column_types()[0]
        if given_type != existing_type:
            raise TypeError("Type of given values does not match type of column '" +
                column_name + "' in SFrame.")

        # Make sure the values list has unique values, or else join will not
        # filter.
        value_sf = value_sf.groupby(column_name, {})

        with cython_context():
            if exclude:
                id_name = "id"
                # Make sure this name is unique so we know what to remove in
                # the result
                while id_name in existing_columns:
                    id_name += "1"
                value_sf = value_sf.add_row_number(id_name)

                tmp = SFrame(_proxy=self.__proxy__.join(value_sf.__proxy__,
                                                     'left',
                                                     {column_name:column_name}))
                ret_sf = tmp[tmp[id_name] == None]
                del ret_sf[id_name]
                return ret_sf
            else:
                return SFrame(_proxy=self.__proxy__.join(value_sf.__proxy__,
                                                     'inner',
                                                     {column_name:column_name}))

    def explore(self, title=None):
        """
        Explore the SFrame in an interactive GUI. Opens a new app window.

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
        Suppose 'sf' is an SFrame, we can view it using:

        >>> sf.explore()

        To override the default plot title and axis labels:

        >>> sf.explore(title="My Plot Title")
        """

        import sys
        import os

        if sys.platform != 'darwin' and sys.platform != 'linux2' and sys.platform != 'linux':
            raise NotImplementedError('Visualization is currently supported only on macOS and Linux.')

        path_to_client = _get_client_app_path()

        if title is None:
            title = ""
        self.__proxy__.explore(path_to_client, title)

    def show(self):
        """
        Visualize a summary of each column in an SFrame. Opens a new app window.

        Notes
        -----
        - The plot will render either inline in a Jupyter Notebook, or in a
          native GUI window, depending on the value provided in
          `turicreate.visualization.set_target` (defaults to 'auto').

        Returns
        -------
        None

        Examples
        --------
        Suppose 'sf' is an SFrame, we can view it using:

        >>> sf.show()
        """

        returned_plot = self.plot()

        returned_plot.show()

    def plot(self):
        """
        Create a Plot object that contains a summary of each column 
        in an SFrame. 

        Notes
        -----
        - The plot will render either inline in a Jupyter Notebook, or in a
          native GUI window, depending on the value provided in
          `turicreate.visualization.set_target` (defaults to 'auto').

        Returns
        -------
        out : Plot
        A :class: Plot object that is the columnwise summary of the sframe.

        Examples
        --------
        Suppose 'sf' is an SFrame, we can make a plot object as:

        >>> plt = sf.plot()

        We can then visualize the plot using:

        >>> plt.show()
        """
        path_to_client = _get_client_app_path()

        return Plot(self.__proxy__.plot(path_to_client))

    def pack_columns(self, column_names=None, column_name_prefix=None, dtype=list,
                     fill_na=None, remove_prefix=True, new_column_name=None):
        """
        Pack columns of the current SFrame into one single column. The result
        is a new SFrame with the unaffected columns from the original SFrame
        plus the newly created column.

        The list of columns that are packed is chosen through either the
        ``column_names`` or ``column_name_prefix`` parameter. Only one of the parameters
        is allowed to be provided. ``columns_names`` explicitly specifies the list of
        columns to pack, while ``column_name_prefix`` specifies that all columns that
        have the given prefix are to be packed.

        The type of the resulting column is decided by the ``dtype`` parameter.
        Allowed values for ``dtype`` are dict, array.array and list:

         - *dict*: pack to a dictionary SArray where column name becomes
           dictionary key and column value becomes dictionary value

         - *array.array*: pack all values from the packing columns into an array

         - *list*: pack all values from the packing columns into a list.

        Parameters
        ----------
        column_names : list[str], optional
            A list of column names to be packed.  If omitted and
            `column_name_prefix` is not specified, all columns from current SFrame
            are packed.  This parameter is mutually exclusive with the
            `column_name_prefix` parameter.

        column_name_prefix : str, optional
            Pack all columns with the given `column_name_prefix`.
            This parameter is mutually exclusive with the `columns_names` parameter.

        dtype : dict | array.array | list, optional
            The resulting packed column type. If not provided, dtype is list.

        fill_na : value, optional
            Value to fill into packed column if missing value is encountered.
            If packing to dictionary, `fill_na` is only applicable to dictionary
            values; missing keys are not replaced.

        remove_prefix : bool, optional
            If True and `column_name_prefix` is specified, the dictionary key will
            be constructed by removing the prefix from the column name.
            This option is only applicable when packing to dict type.

        new_column_name : str, optional
            Packed column name.  If not given and `column_name_prefix` is given,
            then the prefix will be used as the new column name, otherwise name
            is generated automatically.

        Returns
        -------
        out : SFrame
            An SFrame that contains columns that are not packed, plus the newly
            packed column.

        See Also
        --------
        unpack

        Notes
        -----
        - If packing to dictionary, missing key is always dropped. Missing
          values are dropped if fill_na is not provided, otherwise, missing
          value is replaced by 'fill_na'. If packing to list or array, missing
          values will be kept. If 'fill_na' is provided, the missing value is
          replaced with 'fill_na' value.

        Examples
        --------
        Suppose 'sf' is an an SFrame that maintains business category
        information:

        >>> sf = turicreate.SFrame({'business': range(1, 5),
        ...                       'category.retail': [1, None, 1, None],
        ...                       'category.food': [1, 1, None, None],
        ...                       'category.service': [None, 1, 1, None],
        ...                       'category.shop': [1, 1, None, 1]})
        >>> sf
        +----------+-----------------+---------------+------------------+---------------+
        | business | category.retail | category.food | category.service | category.shop |
        +----------+-----------------+---------------+------------------+---------------+
        |    1     |        1        |       1       |       None       |       1       |
        |    2     |       None      |       1       |        1         |       1       |
        |    3     |        1        |      None     |        1         |      None     |
        |    4     |       None      |       1       |       None       |       1       |
        +----------+-----------------+---------------+------------------+---------------+
        [4 rows x 5 columns]

        To pack all category columns into a list:

        >>> sf.pack_columns(column_name_prefix='category')
        +----------+-----------------------+
        | business |        category       |
        +----------+-----------------------+
        |    1     |    [1, 1, None, 1]    |
        |    2     |    [1, None, 1, 1]    |
        |    3     |   [None, 1, 1, None]  |
        |    4     | [None, None, None, 1] |
        +----------+-----------------------+
        [4 rows x 2 columns]

        To pack all category columns into a dictionary, with new column name:

        >>> sf.pack_columns(column_name_prefix='category', dtype=dict,
        ...                 new_column_name='new name')
        +----------+-------------------------------+
        | business |            new name           |
        +----------+-------------------------------+
        |    1     | {'food': 1, 'shop': 1, 're... |
        |    2     | {'food': 1, 'shop': 1, 'se... |
        |    3     |  {'retail': 1, 'service': 1}  |
        |    4     |          {'shop': 1}          |
        +----------+-------------------------------+
        [4 rows x 2 columns]

        To keep column prefix in the resulting dict key:

        >>> sf.pack_columns(column_name_prefix='category', dtype=dict,
                            remove_prefix=False)
        +----------+-------------------------------+
        | business |            category           |
        +----------+-------------------------------+
        |    1     | {'category.retail': 1, 'ca... |
        |    2     | {'category.food': 1, 'cate... |
        |    3     | {'category.retail': 1, 'ca... |
        |    4     |      {'category.shop': 1}     |
        +----------+-------------------------------+
        [4 rows x 2 columns]

        To explicitly pack a set of columns:

        >>> sf.pack_columns(column_names = ['business', 'category.retail',
                                       'category.food', 'category.service',
                                       'category.shop'])
        +-----------------------+
        |           X1          |
        +-----------------------+
        |   [1, 1, 1, None, 1]  |
        |   [2, None, 1, 1, 1]  |
        | [3, 1, None, 1, None] |
        | [4, None, 1, None, 1] |
        +-----------------------+
        [4 rows x 1 columns]

        To pack all columns with name starting with 'category' into an array
        type, and with missing value replaced with 0:

        >>> import array
        >>> sf.pack_columns(column_name_prefix="category", dtype=array.array,
        ...                 fill_na=0)
        +----------+----------------------+
        | business |       category       |
        +----------+----------------------+
        |    1     | [1.0, 1.0, 0.0, 1.0] |
        |    2     | [1.0, 0.0, 1.0, 1.0] |
        |    3     | [0.0, 1.0, 1.0, 0.0] |
        |    4     | [0.0, 0.0, 0.0, 1.0] |
        +----------+----------------------+
        [4 rows x 2 columns]
        """

        if column_names is not None and column_name_prefix is not None:
            raise ValueError("'column_names' and 'column_name_prefix' parameter cannot be given at the same time.")

        if new_column_name is None and column_name_prefix is not None:
            new_column_name = column_name_prefix

        if column_name_prefix is not None:
            if type(column_name_prefix) != str:
                raise TypeError("'column_name_prefix' must be a string")
            column_names = [name for name in self.column_names() if name.startswith(column_name_prefix)]
            if len(column_names) == 0:
                raise ValueError("There is no column starts with prefix '" + column_name_prefix + "'")
        elif column_names is None:
            column_names = self.column_names()
        else:
            if not _is_non_string_iterable(column_names):
                raise TypeError("column_names must be an iterable type")

            column_name_set = set(self.column_names())
            for column in column_names:
                if (column not in column_name_set):
                    raise ValueError("Current SFrame has no column called '" + str(column) + "'.")

            # check duplicate names
            if len(set(column_names)) != len(column_names):
                raise ValueError("There is duplicate column names in column_names parameter")

        if (dtype not in (dict, list, array.array)):
            raise ValueError("Resulting dtype has to be one of dict/array.array/list type")

        # fill_na value for array needs to be numeric
        if dtype == array.array:
            if (fill_na is not None) and (type(fill_na) not in (int, float)):
                raise ValueError("fill_na value for array needs to be numeric type")
            # all column_names have to be numeric type
            for column in column_names:
                if self[column].dtype not in (int, float):
                    raise TypeError("Column '" + column + "' type is not numeric, cannot pack into array type")

        # generate dict key names if pack to dictionary
        # we try to be smart here
        # if all column names are like: a.b, a.c, a.d,...
        # we then use "b", "c", "d", etc as the dictionary key during packing
        if (dtype == dict) and (column_name_prefix is not None) and (remove_prefix == True):
            size_prefix = len(column_name_prefix)
            first_char = set([c[size_prefix:size_prefix+1] for c in column_names])
            if ((len(first_char) == 1) and first_char.pop() in ['.','-','_']):
                dict_keys = [name[size_prefix+1:] for name in column_names]
            else:
                dict_keys = [name[size_prefix:] for name in column_names]

        else:
            dict_keys = column_names

        rest_columns = [name for name in self.column_names() if name not in column_names]
        if new_column_name is not None:
            if type(new_column_name) != str:
                raise TypeError("'new_column_name' has to be a string")
            if new_column_name in rest_columns:
                raise KeyError("Current SFrame already contains a column name " + new_column_name)
        else:
            new_column_name = ""


        ret_sa = None
        with cython_context():
            ret_sa = SArray(_proxy=self.__proxy__.pack_columns(column_names, dict_keys,
                                                               dtype, fill_na))

        new_sf = self.select_columns(rest_columns)
        new_sf.add_column(ret_sa, new_column_name, inplace=True)
        return new_sf


    def split_datetime(self, column_name, column_name_prefix=None, limit=None, timezone=False):
        """
        Splits a datetime column of SFrame to multiple columns, with each value in a
        separate column. Returns a new SFrame with the expanded column replaced with
        a list of new columns. The expanded column must be of datetime type.

        For more details regarding name generation and
        other, refer to :py:func:`turicreate.SArray.split_datetime()`

        Parameters
        ----------
        column_name : str
            Name of the unpacked column.

        column_name_prefix : str, optional
            If provided, expanded column names would start with the given prefix.
            If not provided, the default value is the name of the expanded column.

        limit: list[str], optional
            Limits the set of datetime elements to expand.
            Possible values are 'year','month','day','hour','minute','second',
            'weekday', 'isoweekday', 'tmweekday', and 'us'.
            If not provided, only ['year','month','day','hour','minute','second']
            are expanded.

        timezone : bool, optional
            A boolean parameter that determines whether to show the timezone
            column or not. Defaults to False.

        Returns
        -------
        out : SFrame
            A new SFrame that contains rest of columns from original SFrame with
            the given column replaced with a collection of expanded columns.

        Examples
        ---------

        >>> sf
        Columns:
            id   int
            submission  datetime
        Rows: 2
        Data:
            +----+-------------------------------------------------+
            | id |               submission                        |
            +----+-------------------------------------------------+
            | 1  | datetime(2011, 1, 21, 7, 17, 21, tzinfo=GMT(+1))|
            | 2  | datetime(2011, 1, 21, 5, 43, 21, tzinfo=GMT(+1))|
            +----+-------------------------------------------------+

        >>> sf.split_datetime('submission',limit=['hour','minute'])
        Columns:
            id  int
            submission.hour int
            submission.minute int
        Rows: 2
        Data:
        +----+-----------------+-------------------+
        | id | submission.hour | submission.minute |
        +----+-----------------+-------------------+
        | 1  |        7        |        17         |
        | 2  |        5        |        43         |
        +----+-----------------+-------------------+
        """
        if column_name not in self.column_names():
            raise KeyError("column '" + column_name + "' does not exist in current SFrame")

        if column_name_prefix is None:
            column_name_prefix = column_name

        new_sf = self[column_name].split_datetime(column_name_prefix, limit, timezone)

        # construct return SFrame, check if there is conflict
        rest_columns =  [name for name in self.column_names() if name != column_name]
        new_names = new_sf.column_names()
        while set(new_names).intersection(rest_columns):
            new_names = [name + ".1" for name in new_names]
        new_sf.rename(dict(list(zip(new_sf.column_names(), new_names))), inplace=True)

        ret_sf = self.select_columns(rest_columns)
        ret_sf.add_columns(new_sf, inplace=True)
        return ret_sf

    def unpack(self, column_name, column_name_prefix=None, column_types=None,
               na_value=None, limit=None):
        """
        Expand one column of this SFrame to multiple columns with each value in
        a separate column. Returns a new SFrame with the unpacked column
        replaced with a list of new columns.  The column must be of
        list/array/dict type.

        For more details regarding name generation, missing value handling and
        other, refer to the SArray version of
        :py:func:`~turicreate.SArray.unpack()`.

        Parameters
        ----------
        column_name : str
            Name of the unpacked column

        column_name_prefix : str, optional
            If provided, unpacked column names would start with the given
            prefix. If not provided, default value is the name of the unpacked
            column.

        column_types : [type], optional
            Column types for the unpacked columns.
            If not provided, column types are automatically inferred from first
            100 rows. For array type, default column types are float.  If
            provided, column_types also restricts how many columns to unpack.

        na_value : flexible_type, optional
            If provided, convert all values that are equal to "na_value" to
            missing value (None).

        limit : list[str] | list[int], optional
            Control unpacking only a subset of list/array/dict value. For
            dictionary SArray, `limit` is a list of dictionary keys to restrict.
            For list/array SArray, `limit` is a list of integers that are
            indexes into the list/array value.

        Returns
        -------
        out : SFrame
            A new SFrame that contains rest of columns from original SFrame with
            the given column replaced with a collection of unpacked columns.

        See Also
        --------
        pack_columns, SArray.unpack

        Examples
        ---------
        >>> sf = turicreate.SFrame({'id': [1,2,3],
        ...                      'wc': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        +----+------------------+
        | id |        wc        |
        +----+------------------+
        | 1  |     {'a': 1}     |
        | 2  |     {'b': 2}     |
        | 3  | {'a': 1, 'b': 2} |
        +----+------------------+
        [3 rows x 2 columns]

        >>> sf.unpack('wc')
        +----+------+------+
        | id | wc.a | wc.b |
        +----+------+------+
        | 1  |  1   | None |
        | 2  | None |  2   |
        | 3  |  1   |  2   |
        +----+------+------+
        [3 rows x 3 columns]

        To not have prefix in the generated column name:

        >>> sf.unpack('wc', column_name_prefix="")
        +----+------+------+
        | id |  a   |  b   |
        +----+------+------+
        | 1  |  1   | None |
        | 2  | None |  2   |
        | 3  |  1   |  2   |
        +----+------+------+
        [3 rows x 3 columns]

        To limit subset of keys to unpack:

        >>> sf.unpack('wc', limit=['b'])
        +----+------+
        | id | wc.b |
        +----+------+
        | 1  | None |
        | 2  |  2   |
        | 3  |  2   |
        +----+------+
        [3 rows x 3 columns]

        To unpack an array column:

        >>> import array
        >>> sf = turicreate.SFrame({'id': [1,2,3],
        ...                       'friends': [array.array('d', [1.0, 2.0, 3.0]),
        ...                                   array.array('d', [2.0, 3.0, 4.0]),
        ...                                   array.array('d', [3.0, 4.0, 5.0])]})
        >>> sf
        +-----------------+----+
        |     friends     | id |
        +-----------------+----+
        | [1.0, 2.0, 3.0] | 1  |
        | [2.0, 3.0, 4.0] | 2  |
        | [3.0, 4.0, 5.0] | 3  |
        +-----------------+----+
        [3 rows x 2 columns]

        >>> sf.unpack('friends')
        +----+-----------+-----------+-----------+
        | id | friends.0 | friends.1 | friends.2 |
        +----+-----------+-----------+-----------+
        | 1  |    1.0    |    2.0    |    3.0    |
        | 2  |    2.0    |    3.0    |    4.0    |
        | 3  |    3.0    |    4.0    |    5.0    |
        +----+-----------+-----------+-----------+
        [3 rows x 4 columns]
        """
        if column_name not in self.column_names():
            raise KeyError("column '" + column_name + "' does not exist in current SFrame")

        if column_name_prefix is None:
            column_name_prefix = column_name

        new_sf = self[column_name].unpack(column_name_prefix, column_types, na_value, limit)

        # construct return SFrame, check if there is conflict
        rest_columns =  [name for name in self.column_names() if name != column_name]
        new_names = new_sf.column_names()
        while set(new_names).intersection(rest_columns):
            new_names = [name + ".1" for name in new_names]
        new_sf.rename(dict(list(zip(new_sf.column_names(), new_names))), inplace=True)

        ret_sf = self.select_columns(rest_columns)
        ret_sf.add_columns(new_sf, inplace=True)
        return ret_sf

    def stack(self, column_name, new_column_name=None, drop_na=False, new_column_type=None):
        """
        Convert a "wide" column of an SFrame to one or two "tall" columns by
        stacking all values.

        The stack works only for columns of dict, list, or array type.  If the
        column is dict type, two new columns are created as a result of
        stacking: one column holds the key and another column holds the value.
        The rest of the columns are repeated for each key/value pair.

        If the column is array or list type, one new column is created as a
        result of stacking. With each row holds one element of the array or list
        value, and the rest columns from the same original row repeated.

        The returned SFrame includes the newly created column(s) and all
        columns other than the one that is stacked.

        Parameters
        --------------
        column_name : str
            The column to stack. This column must be of dict/list/array type

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
            A new SFrame that contains newly stacked column(s) plus columns in
            original SFrame other than the stacked column.

        See Also
        --------
        unstack

        Examples
        ---------
        Suppose 'sf' is an SFrame that contains a column of dict type:

        >>> sf = turicreate.SFrame({'topic':[1,2,3,4],
        ...                       'words': [{'a':3, 'cat':2},
        ...                                 {'a':1, 'the':2},
        ...                                 {'the':1, 'dog':3},
        ...                                 {}]
        ...                      })
        +-------+----------------------+
        | topic |        words         |
        +-------+----------------------+
        |   1   |  {'a': 3, 'cat': 2}  |
        |   2   |  {'a': 1, 'the': 2}  |
        |   3   | {'the': 1, 'dog': 3} |
        |   4   |          {}          |
        +-------+----------------------+
        [4 rows x 2 columns]

        Stack would stack all keys in one column and all values in another
        column:

        >>> sf.stack('words', new_column_name=['word', 'count'])
        +-------+------+-------+
        | topic | word | count |
        +-------+------+-------+
        |   1   |  a   |   3   |
        |   1   | cat  |   2   |
        |   2   |  a   |   1   |
        |   2   | the  |   2   |
        |   3   | the  |   1   |
        |   3   | dog  |   3   |
        |   4   | None |  None |
        +-------+------+-------+
        [7 rows x 3 columns]

        Observe that since topic 4 had no words, an empty row is inserted.
        To drop that row, set drop_na=True in the parameters to stack.

        Suppose 'sf' is an SFrame that contains a user and his/her friends,
        where 'friends' columns is an array type. Stack on 'friends' column
        would create a user/friend list for each user/friend pair:

        >>> sf = turicreate.SFrame({'topic':[1,2,3],
        ...                       'friends':[[2,3,4], [5,6],
        ...                                  [4,5,10,None]]
        ...                      })
        >>> sf
        +-------+------------------+
        | topic |     friends      |
        +-------+------------------+
        |  1    |     [2, 3, 4]    |
        |  2    |      [5, 6]      |
        |  3    | [4, 5, 10, None] |
        +----- -+------------------+
        [3 rows x 2 columns]

        >>> sf.stack('friends', new_column_name='friend')
        +-------+--------+
        | topic | friend |
        +-------+--------+
        |   1   |   2    |
        |   1   |   3    |
        |   1   |   4    |
        |   2   |   5    |
        |   2   |   6    |
        |   3   |   4    |
        |   3   |   5    |
        |   3   |   10   |
        |   3   |  None  |
        +-------+--------+
        [9 rows x 2 columns]

        """
        # validate column_name
        column_name = str(column_name)
        if column_name not in self.column_names():
            raise ValueError("Cannot find column '" + str(column_name) + "' in the SFrame.")

        stack_column_type =  self[column_name].dtype
        if (stack_column_type not in [dict, array.array, list]):
            raise TypeError("Stack is only supported for column of dict/list/array type.")

        # user defined types. do some checking
        if new_column_type is not None:
            # if new_column_type is a single type, just make it a list of one type
            if type(new_column_type) is type:
                new_column_type = [new_column_type]

            if (stack_column_type in [list, array.array]) and len(new_column_type) != 1:
                raise ValueError("Expecting a single column type to unpack list or array columns")

            if (stack_column_type in [dict]) and len(new_column_type) != 2:
                raise ValueError("Expecting two column types to unpack a dict column")

        if (new_column_name is not None):
            if stack_column_type == dict:
                if (type(new_column_name) is not list):
                    raise TypeError("new_column_name has to be a list to stack dict type")
                elif (len(new_column_name) != 2):
                    raise TypeError("new_column_name must have length of two")
            else:
                if (type(new_column_name) != str):
                    raise TypeError("new_column_name has to be a str")
                new_column_name = [new_column_name]

            # check if the new column name conflicts with existing ones
            for name in new_column_name:
                if (name in self.column_names()) and (name != column_name):
                    raise ValueError("Column with name '" + name + "' already exists, pick a new column name")
        else:
            if stack_column_type == dict:
                new_column_name = ["",""]
            else:
                new_column_name = [""]

        # infer column types
        head_row = SArray(self[column_name].head(100)).dropna()
        if (len(head_row) == 0):
            raise ValueError("Cannot infer column type because there is not enough rows to infer value")

        if new_column_type is None:
            # we have to perform type inference
            if stack_column_type == dict:
                # infer key/value type
                keys = []; values = []
                for row in head_row:
                    for val in row:
                        keys.append(val)
                        if val is not None: values.append(row[val])

                new_column_type = [
                    infer_type_of_list(keys),
                    infer_type_of_list(values)
                ]
            else:
                values = [v for v in itertools.chain.from_iterable(head_row)]
                new_column_type = [infer_type_of_list(values)]


        with cython_context():
            return SFrame(_proxy=self.__proxy__.stack(column_name,
                                                      new_column_name,
                                                      new_column_type, drop_na))

    def unstack(self, column_names, new_column_name=None):
        """
        Concatenate values from one or two columns into one column, grouping by
        all other columns. The resulting column could be of type list, array or
        dictionary.  If ``column_names`` is a numeric column, the result will be of
        array.array type.  If ``column_names`` is a non-numeric column, the new column
        will be of list type. If ``column_names`` is a list of two columns, the new
        column will be of dict type where the keys are taken from the first
        column in the list.

        Parameters
        ----------
        column_names : str | [str, str]
            The column(s) that is(are) to be concatenated.
            If str, then collapsed column type is either array or list.
            If [str, str], then collapsed column type is dict

        new_column_name : str, optional
            New column name. If not given, a name is generated automatically.

        Returns
        -------
        out : SFrame
            A new SFrame containing the grouped columns as well as the new
            column.

        See Also
        --------
        stack : The inverse of unstack.

        groupby : ``unstack`` is a special version of ``groupby`` that uses the
          :mod:`~turicreate.aggregate.CONCAT` aggregator

        Notes
        -----
        - There is no guarantee the resulting SFrame maintains the same order as
          the original SFrame.

        - Missing values are maintained during unstack.

        - When unstacking into a dictionary, if there is more than one instance
          of a given key for a particular group, an arbitrary value is selected.

        Examples
        --------
        >>> sf = turicreate.SFrame({'count':[4, 2, 1, 1, 2, None],
        ...                       'topic':['cat', 'cat', 'dog', 'elephant', 'elephant', 'fish'],
        ...                       'word':['a', 'c', 'c', 'a', 'b', None]})
        >>> sf.unstack(column_names=['word', 'count'], new_column_name='words')
        +----------+------------------+
        |  topic   |      words       |
        +----------+------------------+
        | elephant | {'a': 1, 'b': 2} |
        |   dog    |     {'c': 1}     |
        |   cat    | {'a': 4, 'c': 2} |
        |   fish   |       None       |
        +----------+------------------+
        [4 rows x 2 columns]

        >>> sf = turicreate.SFrame({'friend': [2, 3, 4, 5, 6, 4, 5, 2, 3],
        ...                      'user': [1, 1, 1, 2, 2, 2, 3, 4, 4]})
        >>> sf.unstack('friend', new_column_name='new name')
        +------+-----------+
        | user |  new name |
        +------+-----------+
        |  3   |    [5]    |
        |  1   | [2, 3, 4] |
        |  2   | [6, 4, 5] |
        |  4   |   [2, 3]  |
        +------+-----------+
        [4 rows x 2 columns]
        """
        if (type(column_names) != str and len(column_names) != 2):
            raise TypeError("'column_names' parameter has to be either a string or a list of two strings.")

        with cython_context():
            if type(column_names) == str:
                key_columns = [i for i in self.column_names() if i != column_names]
                if new_column_name is not None:
                    return self.groupby(key_columns, {new_column_name : aggregate.CONCAT(column_names)})
                else:
                    return self.groupby(key_columns, aggregate.CONCAT(column_names))
            elif len(column_names) == 2:
                key_columns = [i for i in self.column_names() if i not in column_names]
                if new_column_name is not None:
                    return self.groupby(key_columns, {new_column_name: aggregate.CONCAT(column_names[0], column_names[1])})
                else:
                    return self.groupby(key_columns, aggregate.CONCAT(column_names[0], column_names[1]))

    def unique(self):
        """
        Remove duplicate rows of the SFrame. Will not necessarily preserve the
        order of the given SFrame in the new SFrame.

        Returns
        -------
        out : SFrame
            A new SFrame that contains the unique rows of the current SFrame.

        Raises
        ------
        TypeError
          If any column in the SFrame is a dictionary type.

        See Also
        --------
        SArray.unique

        Examples
        --------
        >>> sf = turicreate.SFrame({'id':[1,2,3,3,4], 'value':[1,2,3,3,4]})
        >>> sf
        +----+-------+
        | id | value |
        +----+-------+
        | 1  |   1   |
        | 2  |   2   |
        | 3  |   3   |
        | 3  |   3   |
        | 4  |   4   |
        +----+-------+
        [5 rows x 2 columns]

        >>> sf.unique()
        +----+-------+
        | id | value |
        +----+-------+
        | 2  |   2   |
        | 4  |   4   |
        | 3  |   3   |
        | 1  |   1   |
        +----+-------+
        [4 rows x 2 columns]
        """
        return self.groupby(self.column_names(),{})

    def sort(self, key_column_names, ascending=True):
        """
        Sort current SFrame by the given columns, using the given sort order.
        Only columns that are type of str, int and float can be sorted.

        Parameters
        ----------
        key_column_names : str | list of str | list of (str, bool) pairs
            Names of columns to be sorted.  The result will be sorted first by
            first column, followed by second column, and so on. All columns will
            be sorted in the same order as governed by the `ascending`
            parameter. To control the sort ordering for each column
            individually, `key_column_names` must be a list of (str, bool) pairs.
            Given this case, the first value is the column name and the second
            value is a boolean indicating whether the sort order is ascending.

        ascending : bool, optional
            Sort all columns in the given order.

        Returns
        -------
        out : SFrame
            A new SFrame that is sorted according to given sort criteria

        See Also
        --------
        topk

        Examples
        --------
        Suppose 'sf' is an sframe that has three columns 'a', 'b', 'c'.
        To sort by column 'a', ascending

        >>> sf = turicreate.SFrame({'a':[1,3,2,1],
        ...                       'b':['a','c','b','b'],
        ...                       'c':['x','y','z','y']})
        >>> sf
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | a | x |
        | 3 | c | y |
        | 2 | b | z |
        | 1 | b | y |
        +---+---+---+
        [4 rows x 3 columns]

        >>> sf.sort('a')
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | a | x |
        | 1 | b | y |
        | 2 | b | z |
        | 3 | c | y |
        +---+---+---+
        [4 rows x 3 columns]

        To sort by column 'a', descending

        >>> sf.sort('a', ascending = False)
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 3 | c | y |
        | 2 | b | z |
        | 1 | a | x |
        | 1 | b | y |
        +---+---+---+
        [4 rows x 3 columns]

        To sort by column 'a' and 'b', all ascending

        >>> sf.sort(['a', 'b'])
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | a | x |
        | 1 | b | y |
        | 2 | b | z |
        | 3 | c | y |
        +---+---+---+
        [4 rows x 3 columns]

        To sort by column 'a' ascending, and then by column 'c' descending

        >>> sf.sort([('a', True), ('c', False)])
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | b | y |
        | 1 | a | x |
        | 2 | b | z |
        | 3 | c | y |
        +---+---+---+
        [4 rows x 3 columns]
        """
        sort_column_names = []
        sort_column_orders = []

        # validate key_column_names
        if (type(key_column_names) == str):
            sort_column_names = [key_column_names]
        elif (type(key_column_names) == list):
            if (len(key_column_names) == 0):
                raise ValueError("Please provide at least one column to sort")

            first_param_types = set([type(i) for i in key_column_names])
            if (len(first_param_types) != 1):
                raise ValueError("key_column_names element are not of the same type")

            first_param_type = first_param_types.pop()
            if (first_param_type == tuple):
                sort_column_names = [i[0] for i in key_column_names]
                sort_column_orders = [i[1] for i in key_column_names]
            elif(first_param_type == str):
                sort_column_names = key_column_names
            else:
                raise TypeError("key_column_names type is not supported")
        else:
            raise TypeError("key_column_names type is not correct. Supported types are str, list of str or list of (str,bool) pair.")

        # use the second parameter if the sort order is not given
        if (len(sort_column_orders) == 0):
            sort_column_orders = [ascending for i in sort_column_names]

        # make sure all column exists
        my_column_names = set(self.column_names())
        for column in sort_column_names:
            if (type(column) != str):
                raise TypeError("Only string parameter can be passed in as column names")
            if (column not in my_column_names):
                raise ValueError("SFrame has no column named: '" + str(column) + "'")
            if (self[column].dtype not in (str, int, float,datetime.datetime)):
                raise TypeError("Only columns of type (str, int, float) can be sorted")


        with cython_context():
            return SFrame(_proxy=self.__proxy__.sort(sort_column_names, sort_column_orders))

    def dropna(self, columns=None, how='any'):
        """
        Remove missing values from an SFrame. A missing value is either ``None``
        or ``NaN``.  If ``how`` is 'any', a row will be removed if any of the
        columns in the ``columns`` parameter contains at least one missing
        value.  If ``how`` is 'all', a row will be removed if all of the columns
        in the ``columns`` parameter are missing values.

        If the ``columns`` parameter is not specified, the default is to
        consider all columns when searching for missing values.

        Parameters
        ----------
        columns : list or str, optional
            The columns to use when looking for missing values. By default, all
            columns are used.

        how : {'any', 'all'}, optional
            Specifies whether a row should be dropped if at least one column
            has missing values, or if all columns have missing values.  'any' is
            default.

        Returns
        -------
        out : SFrame
            SFrame with missing values removed (according to the given rules).

        See Also
        --------
        dropna_split :  Drops missing rows from the SFrame and returns them.

        Examples
        --------
        Drop all missing values.

        >>> sf = turicreate.SFrame({'a': [1, None, None], 'b': ['a', 'b', None]})
        >>> sf.dropna()
        +---+---+
        | a | b |
        +---+---+
        | 1 | a |
        +---+---+
        [1 rows x 2 columns]

        Drop rows where every value is missing.

        >>> sf.dropna(any="all")
        +------+---+
        |  a   | b |
        +------+---+
        |  1   | a |
        | None | b |
        +------+---+
        [2 rows x 2 columns]

        Drop rows where column 'a' has a missing value.

        >>> sf.dropna('a', any="all")
        +---+---+
        | a | b |
        +---+---+
        | 1 | a |
        +---+---+
        [1 rows x 2 columns]
        """

        # If the user gives me an empty list (the indicator to use all columns)
        # NA values being dropped would not be the expected behavior. This
        # is a NOOP, so let's not bother the server
        if type(columns) is list and len(columns) == 0:
            return SFrame(_proxy=self.__proxy__)

        (columns, all_behavior) = self.__dropna_errchk(columns, how)

        with cython_context():
            return SFrame(_proxy=self.__proxy__.drop_missing_values(columns, all_behavior, False))

    def dropna_split(self, columns=None, how='any'):
        """
        Split rows with missing values from this SFrame. This function has the
        same functionality as :py:func:`~turicreate.SFrame.dropna`, but returns a
        tuple of two SFrames.  The first item is the expected output from
        :py:func:`~turicreate.SFrame.dropna`, and the second item contains all the
        rows filtered out by the `dropna` algorithm.

        Parameters
        ----------
        columns : list or str, optional
            The columns to use when looking for missing values. By default, all
            columns are used.

        how : {'any', 'all'}, optional
            Specifies whether a row should be dropped if at least one column
            has missing values, or if all columns have missing values.  'any' is
            default.

        Returns
        -------
        out : (SFrame, SFrame)
            (SFrame with missing values removed,
             SFrame with the removed missing values)

        See Also
        --------
        dropna

        Examples
        --------
        >>> sf = turicreate.SFrame({'a': [1, None, None], 'b': ['a', 'b', None]})
        >>> good, bad = sf.dropna_split()
        >>> good
        +---+---+
        | a | b |
        +---+---+
        | 1 | a |
        +---+---+
        [1 rows x 2 columns]

        >>> bad
        +------+------+
        |  a   |  b   |
        +------+------+
        | None |  b   |
        | None | None |
        +------+------+
        [2 rows x 2 columns]
        """

        # If the user gives me an empty list (the indicator to use all columns)
        # NA values being dropped would not be the expected behavior. This
        # is a NOOP, so let's not bother the server
        if type(columns) is list and len(columns) == 0:
            return (SFrame(_proxy=self.__proxy__), SFrame())

        (columns, all_behavior) = self.__dropna_errchk(columns, how)

        sframe_tuple = self.__proxy__.drop_missing_values(columns, all_behavior, True)

        if len(sframe_tuple) != 2:
            raise RuntimeError("Did not return two SFrames!")

        with cython_context():
            return (SFrame(_proxy=sframe_tuple[0]), SFrame(_proxy=sframe_tuple[1]))

    def __dropna_errchk(self, columns, how):
        if columns is None:
            # Default behavior is to consider every column, specified to
            # the server by an empty list (to avoid sending all the column
            # in this case, since it is the most common)
            columns = list()
        elif type(columns) is str:
            columns = [columns]
        elif type(columns) is not list:
            raise TypeError("Must give columns as a list, str, or 'None'")
        else:
            # Verify that we are only passing strings in our list
            list_types = set([type(i) for i in columns])
            if (str not in list_types) or (len(list_types) > 1):
                raise TypeError("All columns must be of 'str' type")


        if how not in ['any','all']:
            raise ValueError("Must specify 'any' or 'all'")

        if how == 'all':
            all_behavior = True
        else:
            all_behavior = False

        return (columns, all_behavior)

    def fillna(self, column_name, value):
        """
        Fill all missing values with a given value in a given column. If the
        ``value`` is not the same type as the values in ``column_name``, this method
        attempts to convert the value to the original column's type. If this
        fails, an error is raised.

        Parameters
        ----------
        column_name : str
            The name of the column to modify.

        value : type convertible to SArray's type
            The value used to replace all missing values.

        Returns
        -------
        out : SFrame
            A new SFrame with the specified value in place of missing values.

        See Also
        --------
        dropna

        Examples
        --------
        >>> sf = turicreate.SFrame({'a':[1, None, None],
        ...                       'b':['13.1', '17.2', None]})
        >>> sf = sf.fillna('a', 0)
        >>> sf
        +---+------+
        | a |  b   |
        +---+------+
        | 1 | 13.1 |
        | 0 | 17.2 |
        | 0 | None |
        +---+------+
        [3 rows x 2 columns]
        """
        # Normal error checking
        if type(column_name) is not str:
            raise TypeError("column_name must be a str")
        ret = self[self.column_names()]
        ret[column_name] = ret[column_name].fillna(value)
        return ret

    def add_row_number(self, column_name='id', start=0, inplace=False):
        """
        Returns an SFrame with a new column that numbers each row
        sequentially. By default the count starts at 0, but this can be changed
        to a positive or negative number.  The new column will be named with
        the given column name.  An error will be raised if the given column
        name already exists in the SFrame.

        If inplace == False (default) this operation does not modify the
        current SFrame, returning a new SFrame.

        If inplace == True, this operation modifies the current
        SFrame, returning self.

        Parameters
        ----------
        column_name : str, optional
            The name of the new column that will hold the row numbers.

        start : int, optional
            The number used to start the row number count.


        inplace : bool, optional. Defaults to False.
            Whether the SFrame is modified in place.

        Returns
        -------
        out : SFrame
            The new SFrame with a column name

        Notes
        -----
        The range of numbers is constrained by a signed 64-bit integer, so
        beware of overflow if you think the results in the row number column
        will be greater than 9 quintillion.

        Examples
        --------
        >>> sf = turicreate.SFrame({'a': [1, None, None], 'b': ['a', 'b', None]})
        >>> sf.add_row_number()
        +----+------+------+
        | id |  a   |  b   |
        +----+------+------+
        | 0  |  1   |  a   |
        | 1  | None |  b   |
        | 2  | None | None |
        +----+------+------+
        [3 rows x 3 columns]
        """

        if type(column_name) is not str:
            raise TypeError("Must give column_name as strs")

        if type(start) is not int:
            raise TypeError("Must give start as int")

        if column_name in self.column_names():
            raise RuntimeError("Column '" + column_name + "' already exists in the current SFrame")

        the_col = _create_sequential_sarray(self.num_rows(), start)

        # Make sure the row number column is the first column
        new_sf = SFrame()
        new_sf.add_column(the_col, column_name, inplace=True)
        new_sf.add_columns(self, inplace=True)

        if inplace:
            self.__proxy__ = new_sf.__proxy__
            return self
        else:
            return new_sf

    def _group(self, key_columns):
        """
        Left undocumented intentionally.
        """
        gsf = GroupedSFrame(self, key_columns)
        return gsf

    @property
    def shape(self):
        """
        The shape of the SFrame, in a tuple. The first entry is the number of
        rows, the second is the number of columns.

        Examples
        --------
        >>> sf = turicreate.SFrame({'id':[1,2,3], 'val':['A','B','C']})
        >>> sf.shape
        (3, 2)
        """
        return (self.num_rows(), self.num_columns())

    @property
    def __proxy__(self):
        return self._proxy

    @__proxy__.setter
    def __proxy__(self, value):
        assert type(value) is UnitySFrameProxy
        self._cache = None
        self._proxy = value
        self._cache = None
