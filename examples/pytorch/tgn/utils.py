# TODO: Implement the index linked list

class DictNode:
    def __init__(self,parent,NIDdict=None):
        self.parent = parent
        if NIDdict != None:
            self.NIDdict = NIDdict.tolist()
        else:
            self.NIDdict = None 
        if (parent!=None and NIDdict==None) or (parent == None and NIDdict!=None):
            raise ValueError("Parent and Dict Unmatched")
            
        #self.child = None

    def GetRootID(self,index):
        if self.parent==None:
            return index
        map_index = self.NIDdict[index]
        return self.parent.GetRootID(map_index)

## tests
import dgl
import torch
g = dgl.graph(([0,1,2,3,4,5],[1,2,3,4,5,0]))
g.ndata['v'] = torch.tensor([1,2,3,4,5,6])
g_d = DictNode(parent=None)

sg = dgl.node_subgraph(g,[1,2,3,4,5])
sg_d = DictNode(parent=g_d,NIDdict=sg.ndata[dgl.NID])

ssg = dgl.node_subgraph(g,[1,2,3,4])
ssg_d = DictNode(parent=sg_d,NIDdict=ssg.ndata[dgl.NID])

id = ssg_d.GetRootID(3)
print(g.ndata['v'][id]) # Expected 6

