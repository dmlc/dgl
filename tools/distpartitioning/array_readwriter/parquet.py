import logging

import pandas as pd
import pyarrow
import pyarrow.parquet

from .registry import register_array_parser


@register_array_parser("parquet")
class ParquetArrayParser(object):
    def __init__(self):
        pass

    def read(self, path):
        logging.info("Reading from %s using parquet format" % path)
        metadata = pyarrow.parquet.read_metadata(path)
        metadata = metadata.schema.to_arrow_schema().metadata
        # As parquet data are tabularized, we assume the dim of ndarray is 2.
        # If not, it should be explictly specified in the file as metadata.
        shape = metadata.get(b"shape", None)
        table = pyarrow.parquet.read_table(path, memory_map=True)
        logging.info("Done reading from %s" % path)
        arr = table.to_pandas().to_numpy()
        if not shape:
            logging.warning(
                "Shape information not found in the metadata, read the data as "
                "a 2 dim array."
            )
        shape = tuple(eval(shape.decode())) if shape else arr.shape
        return arr.reshape(shape)

    def write(self, path, array):
        logging.info("Writing to %s using parquet format" % path)
        shape = array.shape
        if len(shape) > 2:
            array = array.reshape(shape[0], -1)
        table = pyarrow.Table.from_pandas(pd.DataFrame(array))
        table = table.replace_schema_metadata({"shape": str(shape)})
        pyarrow.parquet.write_table(table, path)
        logging.info("Done writing to %s" % path)
