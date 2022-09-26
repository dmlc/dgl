
import dgl
import numpy as np

num_edges = 200 + 1000 + 2000 + 1000

# Create Id Map here.
edge_dict = {
                "author:affiliated_with:institution" : np.array([0, 200]).reshape(1,2),
                "author:writes:paper" : np.array([200, 1200]).reshape(1, 2),
                "paper:cites:paper" : np.array([1200, 3200]).reshape(1, 2),
                "paper:rev_writes:author" : np.array([3200, 4200]).reshape(1, 2)
                }
id_map = dgl.distributed.id_map.IdMap(edge_dict)


orig_eids = np.arange(num_edges)

etype_id, type_eids = id_map(orig_eids)


print('Testing ID MAP')
print('etype_ids: ', np.bincount(etype_id))
