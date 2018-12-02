import networkx as nx
import dgl
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import sparse
from dgl import DGLGraph


"""
# This should be ignored.
"""
def graph_viz(labels,subgraph):
    value = [labels[node] for node in subgraph.nodes()]
    pos = nx.spring_layout(subgraph, seed=1)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    nx.draw_networkx(subgraph,
                     pos=pos,
                     k=1/800,
                     edge_color='k',
                     node_size=50,
                     cmap=plt.get_cmap('coolwarm'),
                     node_color=value,
                     arrows=False,
                     width=0.05,
                     style='dotted',
                     with_labels=False)

def linegraph_accuracy(z_list, labels):
    accu = []
    ybar_list = [th.max(z, 1)[1] for z in z_list]
    for y_bar in ybar_list:
        accuracy = max(th.sum(y_bar == th.LongTensor(label)).item() for label in labels) / len(labels[0])
        accu.append(accuracy)
    return sum(accu) / len(accu)


def linegraph_inference(g, lg, deg_g, deg_lg, pm_pd, feature, label, equi_label, model):
    z = model(g, lg, deg_g, deg_lg, pm_pd)

    z_list = [z]

    equi_labels = [label, equi_label]

    accu = linegraph_accuracy(z_list, equi_labels)

    return accu

def linegraph_inference_viz(g, lg, deg_g, deg_lg, pm_pd, feature, model):
    z = model(g, lg, deg_g, deg_lg, pm_pd)

    z_list = [z]

    ybar_list = [th.max(z, 1)[1] for z in z_list]
    graph_viz(ybar_list[0],g.to_networkx())

def sparse2th(mat):
    value = mat.data
    indices = th.LongTensor([mat.row, mat.col])

    tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), th.Size(mat.shape))

    return tensor

def matrixDebugger(mat, verbose=True):
    print("type is", type(mat))
    print("shape is", mat.shape)
    if verbose:
        print("value is", mat)

def from_npsp(f, *args):
    def wrap(*args):
        new = [th.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args]
        new = [sparse2th(x) if isinstance(x, sparse.coo.coo_matrix) else x for x in new]
        return f(*new)
    return wrap

def check_intra_prob(g, labels, num_classes):
    g_nx = DGLGraph(g).to_networkx()
    adj = nx.adjacency_matrix(g_nx).todense()
    class_to_intra_prob = {i: 0 for i in range(num_classes)}
    class_to_total = {i: 0 for i in range(num_classes)}
    class_to_intra_sum = {i: 0 for i in range(num_classes)}
    for i in range(g_nx.number_of_nodes()):
        _, index = np.where(adj[i] !=0)
        each_total = len(index)
        intra_num = 0
        for j in index:
            if labels[i] == labels[j]:
                intra_num += 1
        class_to_total[int(labels[i])] += each_total
        class_to_intra_sum[int(labels[i])] += intra_num
    for i in range(num_classes):
        class_to_intra_prob[i] = class_to_intra_sum[i] / class_to_total[i]




    for cat in class_to_intra_prob:
        print("Class {} : intra connection probability {}".format(cat, class_to_intra_prob[cat]))

def animate(g, label_history):
    fig2 = plt.figure(figsize=(8,8), dpi=150)
    fig2.clf()
    ax = fig2.subplots()
    g_nx = g.to_networkx()
    pos = nx.spring_layout(g_nx, random_state=1)

    def classify_animate(i):
        ax.cla()
        ax.axis('off')
        ax.set_title("community detection result @ epoch %d" % i)
        value = [label_history[i][node] for node in g_nx.nodes()]
        nx.draw_networkx(g_nx,
                         pos=pos,
                         k=1/800,
                         edge_color='k',
                         node_size=50,
                         cmap=plt.get_cmap('coolwarm'),
                         node_color=value,
                         arrows=False,
                         width=0.05,
                         style='dotted',
                         with_labels=False)

    ani = animation.FuncAnimation(fig2, classify_animate, frames=len(label_history), interval=500)
    ani.save('./animationfromutils.gif', writer='imagemagick')
    plt.show()
