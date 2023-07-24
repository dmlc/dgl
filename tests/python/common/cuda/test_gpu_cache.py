#
#   Copyright (c) 2022 by Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import unittest

import backend as F

import dgl
from utils import parametrize_idtype

D = 5


def generate_graph(idtype, grad=False, add_data=True):
    g = dgl.DGLGraph().to(F.ctx(), dtype=idtype)
    g.add_nodes(10)
    u, v = [], []
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        u.append(0)
        v.append(i)
        u.append(i)
        v.append(9)
    # add a back flow from 9 to 0
    u.append(9)
    v.append(0)
    g.add_edges(u, v)
    if add_data:
        ncol = F.randn((10, D))
        ecol = F.randn((17, D))
        if grad:
            ncol = F.attach_grad(ncol)
            ecol = F.attach_grad(ecol)
        g.ndata["h"] = ncol
        g.edata["l"] = ecol
    return g


@unittest.skipIf(not F.gpu_ctx(), reason="only necessary with GPU")
@parametrize_idtype
def test_gpu_cache(idtype):
    g = generate_graph(idtype)
    cache = dgl.cuda.GPUCache(5, D, idtype)
    h = g.ndata["h"]

    t = 5
    keys = F.arange(0, t, dtype=idtype)
    values, m_idx, m_keys = cache.query(keys)
    m_values = h[F.tensor(m_keys, F.int64)]
    values[F.tensor(m_idx, F.int64)] = m_values
    cache.replace(m_keys, m_values)

    keys = F.arange(3, 8, dtype=idtype)
    values, m_idx, m_keys = cache.query(keys)
    assert m_keys.shape[0] == 3 and m_idx.shape[0] == 3
    m_values = h[F.tensor(m_keys, F.int64)]
    values[F.tensor(m_idx, F.int64)] = m_values
    assert (values != h[F.tensor(keys, F.int64)]).sum().item() == 0
    cache.replace(m_keys, m_values)


if __name__ == "__main__":
    test_gpu_cache(F.int64)
    test_gpu_cache(F.int32)
