import dgl

class DGLHeteroGraph(object):
    def __init__(self, metagraph, number_of_nodes_by_type, edge_connections_by_type,
                 node_frame=None, edge_frame=None):
        super(DGLHeteroGraph, self).__init__()

        self.metagraph = metagraph
        self.number_of_nodes_by_type = number_of_nodes_by_type
        self.edge_connections_by_type = edge_connections_by_type
        self.node_frame = node_frame
        self.edge_frame = edge_frame

