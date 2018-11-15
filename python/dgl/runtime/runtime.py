"""DGL mini-runtime."""
from abc import abstractmethod
from collections import namedtuple

class Runtime(object):
    @staticmethod
    def run(execs):
        for exe in execs:
            exe.run()

class VarType(object):
    FEAT = 0
    FEAT_DICT = 1
    SPMAT = 2
    STR = 3

Var = namedtuple('Var', ['name', 'type'])

class InstType(object):
    NODE_UDF = 0
    EDGE_UDF = 1
    SPMV = 2
    SPMV_WITH_DATA = 3
    READ_DICT = 4
    WRITE_DICT = 5
    MERGE = 6

Inst = namedtuple('Inst', ['type', 'args', 'ret'])

"""
env={'u', 'v'}
[snr(u, v, mfn1, rfn1),   # v2v
 snr(u, v, mfn2, rfn2),   # e2v
 snr(u, v, mfn3, rfn3),]  # udf

=> (type)
Feat
FeatDict
SpMat
Str

=> (inst)
spmv_with_dat: SpMat -> Feat -> Feat -> Feat
spmv: SpMat -> Feat -> Feat
edge_udf: FeatDict -> FeatDict -> FeatDict  # for message and apply_edges
node_udf: FeatDict -> FeatDict -> FeatDict  # for apply_nodes and reduce
read: FeatDict -> Str -> Feat
write: FeatDict -> Str -> Feat -> None
merge: FeatDict* -> FeatDict

=>
env, node_frame, edge_frame, g
 SpMat adj = env['adj'] # build_adj_from_edges(u, v),
 SpMat inc = env['inc'] # build_inc_from_nodes(v)

 Feat t1 = read(node_frame, 'x')
 Feat t2 = read(edge_frame, 'w')
 Feat t3 = spmv_with_dat(adj, t1, t2)  # snr1

 FeatDict t4 = node_frame
 FeatDict t5 = edge_frame
 FeatDict t6 = edge_udf(t4, t5)
 Feat t8 = read(t6, 'w')
 Feat t9 = spmv(inc, t8)

 FeatDict t10 = edge_udf(nf, ef)
 FeatDict t11 = node_udf(nf, t10)  # bkt1
 FeatDict t12 = node_udf(nf, t10)  # bkt2
 FeatDict t13 = node_udf(nf, t10)  # bkt3
 FeatDict t14 = node_udf(nf, t10)  # bkt4
 FeatDict t15 = merge([t11, t12, t13, t14])

 write(node_frame, 'y', t3)
 write(node_frame, 'z', t9)
 write_dict(node_frame, t15)
"""
