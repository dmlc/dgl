"""DataPipe Utility Functions"""


def find_parent_dp(graph, dp):
    for k, (cur_dp, v) in graph.items():
        if k == id(dp):
            return v
        else:
            result = find_parent_dp(v, dp)
            if result is not None:
                return result
    return None
