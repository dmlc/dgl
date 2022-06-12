from torch.fx import Node

from ..utils import arg_trace
from ..constants import DGL_FUNCTION, DGL_GRAPH_ATTRIBUTE, DGL_TENSOR_DATA, DGL_VOID_CALL, \
    GET_ATTR

def add_edge(src, dst, allow_break=True):
    edge = GEdge(src, dst, allow_break)
    src.add_out_edge(edge)
    dst.add_in_edge(edge)

def check_allow_break(src, dst):
    if src.op == GET_ATTR:
        return False
    if src.node_type == DGL_GRAPH_ATTRIBUTE and dst.node_type in (DGL_TENSOR_DATA, DGL_VOID_CALL):
        return False
    if src.node_type == DGL_FUNCTION:
        return False
    return True

def get_node_relation(node_list):
    name2gnode_map = {}
    node_relation = []
    for lineno, node in enumerate(node_list):
        node_relation.append(GNode(node, lineno))
        name2gnode_map[node.name] = node_relation[-1]

    for node in node_relation:
        args = arg_trace(node.args)
        for arg in args:
            allow_break = check_allow_break(name2gnode_map[arg.name], node)
            add_edge(name2gnode_map[arg.name], node, allow_break)

    dgl_attr_map = {}
    for node in node_relation:
        if node.node_type == DGL_GRAPH_ATTRIBUTE:
            # TODO double check here
            data_prefix = ""
            # data_prefix = node.args[1]
            for e in node.out_edges:
                dst = e.dst
                if dst.node_type == DGL_VOID_CALL:
                    for data_name, v in dst.args[1].items():
                        # record the item: g.dstdata.update({"x": x})
                        dgl_attr_map[data_prefix + data_name] = dst
                elif dst.node_type == DGL_TENSOR_DATA:
                    data_name = dst.args[1]
                    src_node = dgl_attr_map[data_prefix + data_name]
                    # add link for: x = g.dstdata["x"]
                    add_edge(src_node, node, False)

        elif node.node_type == DGL_FUNCTION:
            # TODO: data prefix
            for k, v in node.kwargs.items():
                if k == "msg_field":
                    update_all_node = dgl_attr_map[v]
                    for e in update_all_node.in_edges:
                        if e.src.node_type == DGL_FUNCTION and k not in e.src.kwargs:
                            add_edge(e.src, node, False)
                            break
                elif "field" in k and k != "out_field":
                    add_edge(dgl_attr_map[v], node, False)
            # dgl function only call once
            assert len(node.out_edges) == 1 and node.out_edges[0].dst.node_type == DGL_VOID_CALL
            dst = node.out_edges[0].dst
            dgl_attr_map[node.kwargs["out_field"]] = dst

    return node_relation


class GNode:
    def __init__(self, node: Node, lineno):
        self.node = node
        self.node_type = node.node_type
        self.name = node.name
        self.op = node.op
        self.args = node.args
        self.kwargs = node.kwargs
        self.target = node.target
        self.lineno = lineno
        self.in_edges = []
        self.out_edges = []

    def add_in_edge(self, e):
        self.in_edges.append(e)

    def add_out_edge(self, e):
        self.out_edges.append(e)

    def __str__(self):
        return "{} {} {}: {} {}".format(self.lineno, self.name, self.node_type, 
            [(("" if e.allow_break else "+") + str(e.src.lineno)) for e in self.in_edges],
            [(("" if e.allow_break else "+") + str(e.dst.lineno)) for e in self.out_edges])

    def __repr__(self):
        return self.name

class GEdge:
    def __init__(self, src: GNode, dst: GNode, allow_break=True):
        self.src = src
        self.dst = dst
        self.allow_break = allow_break

    def __repr(self):
        return "{} - {}".format(self.src.name, self.dst.name)

    def __str__(self):
        return "{} - {}".format(self.src.name, self.dst.name)
