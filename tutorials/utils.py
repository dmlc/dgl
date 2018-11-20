import networkx as nx
import dgl
import torch as th
import numpy as np
import matplotlib.pyplot as plt



def graph_viz(labels,subgraph):
    value = [labels[node] for node in subgraph.nodes()]
    pos = nx.spring_layout(subgraph, random_state=1)
    nx.draw_networkx(subgraph, 
                     pos=pos, 
                     edge_color='k', 
                     node_size=5, 
                     cmap=plt.get_cmap('Set2'), 
                     node_color=value,
                     arrows=False,
                     width=0.6,
                     with_labels=False)

def linegraph_accuracy(z_list, labels):
    accu = []
    ybar_list = [th.max(z, 1)[1] for z in z_list]
    for y_bar in ybar_list:
        accuracy = max(th.sum(y_bar == label).item() for label in labels) / len(labels[0])
        accu.append(accuracy)
    return sum(accu) / len(accu)


def linegraph_inference(g, lg, deg_g, deg_lg, pm_pd, feature, label, equi_label, model):
    z = model(g, lg, deg_g, deg_lg, pm_pd, feature)
    
    z_list = [z]
    
    equi_labels = [label, equi_label]
    
    accu = linegraph_accuracy(z_list, equi_labels)
    
    return accu

def linegraph_inference_viz(g, lg, deg_g, deg_lg, pm_pd, feature, model):
    z = model(g, lg, deg_g, deg_lg, pm_pd, feature)
    
    z_list = [z]
    
    n_batchsize = 1
    ybar_list = [th.max(z, 1)[1] for z in z_list]
    graph_viz(ybar_list[0],g.to_networkx())

def classify_animate(i):
    ax.cla()
    ax.axis('off')
    ax.set_title("community detection result @ epoch %d" % i)
    value = [validation_example_label_change[i][node] for node in g_nx.nodes()]
    nx.draw_networkx(g_nx,
                     pos=pos,
                     edge_color='k',
                     node_size=7,
                     cmap=plt.get_cmap('Set2'),
                     node_color=value,
                     arrows=False,
                     width=0.6,
                     with_labels=False) 


    