from data import TemporalWikipediaDataset, TemporalDataLoader

g = TemporalWikipediaDataset()

# Test temporal dataloader

loader = TemporalDataLoader(g, 200, 20,'topk')

done = False
total_edge = 0
_,_,_,_,_ = loader.get_next_batch()

_,src_l,dst_l,ts,_ = loader.get_next_batch()

for src,dst,t in zip(src_l,dst_l,ts):
    subg = loader.get_edge_affiliation(src,dst,t)
    print(subg)
    

