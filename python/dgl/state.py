"""Our own management of node states.

* It should be compatible with networkx's dict-of-dict-of-dict concept.
* It should use compact tensor storage whenever it is possible.
"""

class NodeDict(dict):
    def register(name, shape, dtype):
        pass

AdjOuterDict = dict
AdjInnerDict = dict

class EdgeAttrDict(dict):
    def register(name, shape, dtype):
        pass
