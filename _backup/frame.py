from dgl.array import DGLArray, DGLDenseArray, DGLSparseArray
import dgl.backend as F
from collections import MutableMapping
from functools import reduce
from itertools import dropwhile
import operator

class DGLFrame(MutableMapping):
    def __init__(self, data=None):
        self._columns = {}
        if data is None:
            pass
        elif isinstance(data, dict):
            for key, value in data.items():
                device = self.device()
                if device:
                    assert value.device() == device
                if type(value) is DGLDenseArray:
                    num_rows = self.num_rows()
                    if num_rows:
                        assert value.shape[0] == num_rows
                self._columns[key] = value
        else:
            raise NotImplementedError()

    def __copy__(self):
        return self._columns.copy()

    def __delitem__(self, key):
        """
        """
        del self._columns[key]

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
            * DGLArray
                Performs a logical filter.  Expects given DGLArray to be the same
                length as all columns in current DGLFrame.  Every row
                corresponding with an entry in the given DGLArray that is
                equivalent to False is filtered from the result.
            * int
                Returns a single row of the DGLFrame (the `key`th one) as a dictionary.
            * slice
                Returns an DGLFrame including only the sliced rows.
        """
        if type(key) is str:
            return self._columns[key]
        elif type(key) is type:
            raise NotImplementedError()
        elif type(key) is list:
            raise NotImplementedError()
        elif type(key) is DGLDenseArray:
            return DGLFrame({k : v[key] for k, v in self._columns.items()})
        elif type(key) is int:
            return {k : v[key] for k, v in self._columns.items()}
        elif type(key) is slice:
            raise NotImplementedError()
        else:
            raise RuntimeError()

    def __iter__(self):
        return iter(self._columns.keys())

    def __len__(self):
        return len(self._columns)

    def __setitem__(self, key, value):
        """
        A wrapper around add_column(s).  Key can be either a list or a str.  If
        value is an DGLArray, it is added to the DGLFrame as a column.  If it is a
        constant value (int, str, or float), then a column is created where
        every entry is equal to the constant value.  Existing columns can also
        be replaced using this wrapper.
        """
        if type(key) is str:
            if type(value) is DGLDenseArray:
                assert value.shape[0] == self.num_rows()
                self._columns[key] = value
            elif type(value) is DGLSparseArray:
                raise NotImplementedError()
            else:
                raise RuntimeError()
        elif type(key) is list:
            raise NotImplementedError()
        else:
            raise RuntimeError()

    def _next_dense_column(self):
        if self._columns:
            predicate = lambda x: type(x) is DGLDenseArray
            try:
                return next(dropwhile(predicate, self._columns.values()))
            except StopIteration:
                return None
        else:
            return None

    def append(self, other):
        """
        Add the rows of an DGLFrame to the end of this DGLFrame.

        Both DGLFrames must have the same set of columns with the same column
        names and column types.

        Parameters
        ----------
        other : DGLFrame
            Another DGLFrame whose rows are appended to the current DGLFrame.

        Returns
        -------
        out : DGLFrame
            The result DGLFrame from the append operation.
        """
        assert isisntance(other, DGLFrame)
        assert set(self._columns) == set(other._columns)
        if self.num_rows() == 0:
            return other.__copy__()
        elif self.num_rows() == 0:
            return self.__copy__()
        else:
            return {k : v.append(other[k]) for k, v in self._columns.items()}

    def device(self):
        dense_column = self._next_dense_column()
        return None if dense_column is None else dense_column.device()

    def dropna(self, columns=None, how='any'):
        columns = list(self._columns) if columns is None else columns

        assert type(columns) is list
        assert len(columns) > 0

        column_list = [self._columns[x] for x in columns]
        if all(type(x) is DGLDenseArray for x in column_list):
            a_list = [x.applicable for x in column_list]
            if how == 'any':
                a = reduce(operator.mul, a_list)
            elif how == 'all':
                a = (reduce(operator.add, a_list) > 0)
            else:
                raise RuntimeError()
            a_array = DGLDenseArray(a)
            return DGLFrame({k : v[a_array] for k, v in self._columns.items()})
        else:
            raise NotImplementedError()

    def filter_by(self, values, column_name, exclude=False):
        """
        Filter an DGLFrame by values inside an iterable object. Result is an
        DGLFrame that only includes (or excludes) the rows that have a column
        with the given ``column_name`` which holds one of the values in the
        given ``values`` :class:`~turicreate.DGLArray`. If ``values`` is not an
        DGLArray, we attempt to convert it to one before filtering.

        Parameters
        ----------
        values : DGLArray | list | numpy.ndarray | pandas.Series | str
            The values to use to filter the DGLFrame.  The resulting DGLFrame will
            only include rows that have one of these values in the given
            column.

        column_name : str
            The column of the DGLFrame to match with the given `values`.

        exclude : bool
            If True, the result DGLFrame will contain all rows EXCEPT those that
            have one of ``values`` in ``column_name``.

        Returns
        -------
        out : DGLFrame
            The filtered DGLFrame.
        """
        if type(values) is DGLDenseArray:
            mask = F.isin(self._columns[column_name], values.data)
            if exclude:
                mask = 1 - mask
            return self[mask]
        else:
            raise NotImplementedError()

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
        out_sf : DGLFrame
            A new DGLFrame, with a column for each groupby column and each
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
        """
        if type(key_column_names) is str:
            if type(operations) is list:
                raise NotImplementedError()
            elif type(operations) is dict:
                if len(operations) == 1:
                    dst_solumn_name, = operations.keys()
                    aggregator, = operations.values()
                    return DGLFrame(aggregator(self, key_column_names))
                else:
                    raise NotImplementedError()
            else:
                raise RuntimeError()
        else:
            raise NotImplementedError()

    def join(self, right, on=None, how='inner'):
        """
        Merge two DGLFrames. Merges the current (left) DGLFrame with the given
        (right) DGLFrame using a SQL-style equi-join operation by columns.

        Parameters
        ----------
        right : DGLFrame
            The DGLFrame to join.

        on : None | str | list | dict, optional
            The column name(s) representing the set of join keys.  Each row that
            has the same value in this set of columns will be merged together.

            * If 'None' is given, join will use all columns that have the same
              name as the set of join keys.

            * If a str is given, this is interpreted as a join using one column,
              where both DGLFrames have the same column name.

            * If a list is given, this is interpreted as a join using one or
              more column names, where each column name given exists in both
              DGLFrames.

            * If a dict is given, each dict key is taken as a column name in the
              left DGLFrame, and each dict value is taken as the column name in
              right DGLFrame that will be joined together. e.g.
              {'left_col_name':'right_col_name'}.

        how : {'left', 'right', 'outer', 'inner'}, optional
            The type of join to perform.  'inner' is default.

            * inner: Equivalent to a SQL inner join.  Result consists of the
              rows from the two frames whose join key values match exactly,
              merged together into one DGLFrame.

            * left: Equivalent to a SQL left outer join. Result is the union
              between the result of an inner join and the rest of the rows from
              the left DGLFrame, merged with missing values.

            * right: Equivalent to a SQL right outer join.  Result is the union
              between the result of an inner join and the rest of the rows from
              the right DGLFrame, merged with missing values.

            * outer: Equivalent to a SQL full outer join. Result is
              the union between the result of a left outer join and a right
              outer join.

        Returns
        -------
        out : DGLFrame
        """
        assert type(right) == DGLFrame
        if on is None:
            raise NotImplementedError()
        elif type(on) is str:
            assert set(self._columns).intersection(set(right._columns)) == {on}
        elif type(on) is list:
            raise NotImplementedError()
        elif type(on) is dict:
            raise NotImplementedError()
        else:
            raise RuntimeError()

        if how == 'left':
            raise NotImplementedError()
        elif how == 'right':
            raise NotImplementedError()
        elif how == 'outer':
            raise NotImplementedError()
        elif how == 'inner':
            lhs = self._columns[on]
            rhs = right._columns[on]
            if type(lhs) is DGLDenseArray and type(rhs) is DGLDenseArray:
                if isinstance(lhs.data, F.Tensor) and isinstance(rhs.data, F.Tensor) and \
                    len(F.shape(lhs.data)) == 1 and len(F.shape(rhs.data)) == 1:
                    assert F.prod(lhs.applicable) and F.prod(rhs.applicable)
                    isin = F.isin(lhs.data, rhs.data)
                    columns = {k : v[isin] for k, v in self._columns.items()}
                    columns.update({k : v for k, v in self._columns.items() if k != on})
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            raise RuntimeError()

    def num_rows(self):
        dense_column = self._next_dense_column()
        return None if dense_column is None else dense_column.shape[0]
