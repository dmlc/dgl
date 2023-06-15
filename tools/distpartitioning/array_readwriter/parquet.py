import logging

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet

from .registry import register_array_parser


@register_array_parser("parquet")
class ParquetArrayParser(object):
    def __init__(self):
        pass

    def read(self, path):
        logging.debug("Reading from %s using parquet format" % path)
        metadata = pyarrow.parquet.read_metadata(path)
        metadata = metadata.schema.to_arrow_schema().metadata

        # As parquet data are tabularized, we assume the dim of ndarray is 2.
        # If not, it should be explictly specified in the file as metadata.
        if metadata:
            shape = metadata.get(b"shape", None)
        else:
            shape = None
        table = pyarrow.parquet.read_table(path, memory_map=True)

        data_types = table.schema.types
        # Spark ML feature processing produces single-column parquet files where each row is a vector object
        if len(data_types) == 1 and isinstance(data_types[0], pyarrow.ListType):
            arr = np.array(table.to_pandas().iloc[:, 0].to_list())
            logging.debug(
                f"Parquet data under {path} converted from single vector per row to ndarray"
            )
        else:
            arr = table.to_pandas().to_numpy()
        if not shape:
            logging.debug(
                "Shape information not found in the metadata, read the data as "
                "a 2 dim array."
            )
        logging.debug("Done reading from %s" % path)
        shape = tuple(eval(shape.decode())) if shape else arr.shape
        return arr.reshape(shape)

    def write(self, path, array, vector_rows=False):
        logging.debug("Writing to %s using parquet format" % path)
        shape = array.shape
        if len(shape) > 2:
            array = array.reshape(shape[0], -1)
        if vector_rows:
            table = pyarrow.table(
                [pyarrow.array(array.tolist())], names=["vector"]
            )
            logging.debug("Writing to %s using single-vector rows..." % path)
        else:
            table = pyarrow.Table.from_pandas(pd.DataFrame(array))
            table = table.replace_schema_metadata({"shape": str(shape)})

        pyarrow.parquet.write_table(table, path)
        logging.debug("Done writing to %s" % path)
