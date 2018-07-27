from dgl.array import DGLArray, DGLDenseArray, DGLSparseArray
import dgl.backend as F

def _gridize(frame, key_column_names, src_column_name):
    if type(key_column_names) is str:
        key_column = frame[key_column_names]
        assert F.prod(key_column.applicable)
        if type(key_column) is DGLDenseArray:
            row = key_column.data
            if type(row) is F.Tensor:
                assert F.isinteger(row) and len(F.shape(row)) == 1
                col = F.unique(row)
                xy = (F.expand_dims(row, 1) == F.expand_dims(col, 0))
                if src_column_name:
                    src_column = frame[src_column_name]
                    if type(src_column) is DGLDenseArray:
                        z = src_column.data
                        if type(z) is F.Tensor:
                            z = F.expand_dims(z, 1)
                            for i in range(2, len(F.shape(z))):
                                xy = F.expand_dims(xy, i)
                            xy = F.astype(xy, F.dtype(z))
                            return col, xy * z
                        elif type(z) is list:
                            raise NotImplementedError()
                        else:
                            raise RuntimeError()
                else:
                    return col, xy
            elif type(row) is list:
                raise NotImplementedError()
            else:
                raise RuntimeError()
        else:
            raise NotImplementedError()
    elif type(key_column_names) is list:
        raise NotImplementedError()
    else:
        raise RuntimeError()

def aggregator(src_column_name=''):
    def decorator(a):
        def decorated(frame, key_column_names):
            col, xy = _gridize(frame, key_column_names, src_column_name)
            trg_column_name = src_column_name + a.__name__
            key = DGLDenseArray(col)
            trg = DGLDenseArray(a(xy))
            return {key_column_names : key, trg_column_name : trg}
        return decorated
    return decorator

def COUNT():
    @aggregator()
    def count(x):
        return F.sum(x, 0)
    return count

def SUM(src_column_name):
    @aggregator(src_column_name)
    def sum(x):
        return F.sum(x, 0)
    return sum
