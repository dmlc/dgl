class Stage(object):
    def __init__(self):
        pass

    def add_executor(self):
        pass


class Executor(object):

    def __init__(self, graph_store):
        pass

    def set_graph_key(self, key):
        pass

    def set_node_input(self, node_frame, node_field=None, node_ids=None):
        pass

    def set_edge_input(self, edge_frame, edge_field=None, edge_ids=None):
        pass

    def set_node_output(self, node_frame, node_field=None, node_ids=None):
        pass

    def set_edge_output(self, edge_frame, edge_field=None, edge_ids=None):
        pass

    def run(self):
        pass

class SPMVExecutor(Executor):
    def __init__(self, src_field):
        self.src_field = src_field

    def custom_run(self, graph_store):
        return {self.dst_field : ...}

class UDFExecutor(Executor):
    def custom_run(self, graph_store):
        return self._fn(...)


class Stages(object):
    def add_stage(self, execs, merge_config):
        me = MergeExecutor(apply_func=...)
        me.execs = execs
        self.execs.extend(execs)
        self.execs.append(me)
