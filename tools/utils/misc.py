import pyarrow
import pandas as pd
import pyarrow.csv
from collections import defaultdict
import os
import copy

def get_cetype_dict_from_meta(edges):
    cetypes = list(edges.keys())
    cetype_tuples = [s.split(':') for s in cetypes]
    cetype_dict = defaultdict(list)
    for utype, etype, vtype in cetype_tuples:
        cetype_dict[etype].append((utype, etype, vtype))
    return cetype_dict

def get_cetype(cetype_dict, etype):
    if ':' in etype:
        return etype.split(':')
    if etype not in cetype_dict:
        raise KeyError('Cannot find edge type %s in metadata' % etype)
    if len(cetype_dict[etype]) > 1:
        raise ValueError(
                'Ambiguous edge type %s (found %s), please specify the full canonical edge type.' % (
                    etype, cetype_dict[etype]))
    return cetype_dict[etype][0]

def join_cetype(cetype):
    if isinstance(cetype, str):
        return cetype
    if ':' in cetype[1]:
        return cetype[1]
    return ':'.join(cetype)

def join_cetype_path(cetype):
    if isinstance(cetype, str):
        return cetype.replace(':', '___')
    if ':' in cetype[1]:
        return cetype[1].replace(':', '___')
    return '___'.join(cetype)

def find_etype_metadata(meta, cetype):
    if cetype in meta:
        return meta[cetype]
    utype, etype, vtype = cetype.split(':')
    if etype in meta:
        return meta[etype]
    meta[cetype] = {}
    return meta[cetype]

def set_etype_metadata(meta, cetype, value):
    if cetype in meta:
        meta[cetype] = value
    utype, etype, vtype = cetype.split(':')
    if etype in meta:
        meta[etype] = value
    meta[cetype] = value

def savetxt(f, arr):
    arr = pyarrow.Table.from_pandas(pd.DataFrame(arr))
    pyarrow.csv.write_csv(arr, f, write_options=pyarrow.csv.WriteOptions(
        delimiter=' ', include_header=False))

def absolutify_metadata(metadata):
    new_metadata = copy.deepcopy(metadata)
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if k == 'path':
                new_metadata[k] = os.path.abspath(v)
            elif isinstance(v, dict):
                new_metadata[k] = absolutify_metadata(v)
    return new_metadata
