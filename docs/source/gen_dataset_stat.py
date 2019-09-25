from pytablewriter import RstGridTableWriter, MarkdownTableWriter
import numpy as np
import pandas as pd
from dgl import DGLGraph
from dgl.data.gnn_benckmark import AmazonCoBuy, CoraFull, Coauthor
from dgl.data.karate import KarateClub
from dgl.data.gindt import GINDataset
from dgl.data.bitcoinotc import BitcoinOTC
from dgl.data.gdelt import GDELT
from dgl.data.icews18 import ICEWS18
from dgl.data.qm7b import QM7b
# from dgl.data.qm9 import QM9
from dgl.data import CitationGraphDataset, CoraDataset, PPIDataset, RedditDataset, TUDataset

ds_list = {
    "BitcoinOTC": "BitcoinOTC()",
    "Cora": "CoraDataset()",
    "Citeseer": "CitationGraphDataset('citeseer')",
    "PubMed": "CitationGraphDataset('pubmed')",
    "QM7b": "QM7b()",
    "Reddit": "RedditDataset()",
    "ENZYMES": "TUDataset('ENZYMES')",
    "DD": "TUDataset('DD')",
    "COLLAB": "TUDataset('COLLAB')",
    "MUTAG": "TUDataset('MUTAG')",
    "PROTEINS": "TUDataset('PROTEINS')",
    "PPI": "PPIDataset('train')/PPIDataset('valid')/PPIDataset('test')",
    # "Cora Binary": "CitationGraphDataset('cora_binary')",
    "KarateClub": "KarateClub()",
    "Amazon computer": "AmazonCoBuy('computers')",
    "Amazon photo": "AmazonCoBuy('photo')",
    "Coauthor cs": "Coauthor('cs')",
    "Coauthor physics": "Coauthor('physics')",
    "GDELT": "GDELT('train')/GDELT('valid')/GDELT('test')",
    "ICEWS18": "ICEWS18('train')/ICEWS18('valid')/ICEWS18('test')",
    "CoraFull": "CoraFull()",
}

writer = RstGridTableWriter()
# writer = MarkdownTableWriter()

extract_graph = lambda g: g if isinstance(g, DGLGraph) else g[0]
stat_list=[]
for k,v in ds_list.items():
    print(k, ' ', v)
    ds = eval(v.split("/")[0])
    num_nodes = []
    num_edges = []
    for i in range(len(ds)):
        g = extract_graph(ds[i])
        num_nodes.append(g.number_of_nodes())
        num_edges.append(g.number_of_edges())

    gg = extract_graph(ds[0])
    dd = {
        "Datset Name": k,
        "Usage": v,
        "# of graphs": len(ds),
        "Avg. # of nodes": np.mean(num_nodes),
        "Avg. # of edges": np.mean(num_edges),
        "Node field": ', '.join(list(gg.ndata.keys())),
        "Edge field": ', '.join(list(gg.edata.keys())),
        # "Graph field": ', '.join(ds[0][0].gdata.keys()) if hasattr(ds[0][0], "gdata") else "",
        "Temporal": hasattr(ds, "is_temporal")
    }
    stat_list.append(dd)

print(dd.keys())
df = pd.DataFrame(stat_list)
df = df.reindex(columns=dd.keys())
writer.from_dataframe(df)

writer.write_table()
