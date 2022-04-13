import numpy as np

def writeNodeFeatures( nodeFeatures, nodeFile ): 
    dgl.data.utils.save_tensors( nodeFile, nodeFeatures )
