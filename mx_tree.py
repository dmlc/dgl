import networkx as nx
import mx

tr = nx.balanced_tree(3, 4)
tr = mx.mx_Graph(tr)

# now we have made a skeleton tree, we can register
# representations to a subset of the nodes, and then
# update them

