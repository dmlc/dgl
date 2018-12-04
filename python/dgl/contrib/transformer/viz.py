import os
import numpy as np
import torch as th
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from networkx.algorithms import bipartite

def get_attention_map(g, src_nodes, dst_nodes, h):
    """
    To visualize the attention score between two set of nodes.
    """
    n, m = len(src_nodes), len(dst_nodes)
    weight = th.zeros(n, m, h).fill_(-1e8)
    for i, src in enumerate(src_nodes.tolist()):
        for j, dst in enumerate(dst_nodes.tolist()):
            if not g.has_edge_between(src, dst):
                continue
            eid = g.edge_id(src, dst)
            weight[i][j] = g.edata['score'][eid].squeeze(-1).cpu().detach()

    weight = weight.transpose(0, 2)
    att = th.softmax(weight, -2)
    return att.numpy()

def draw_heatmap(array, input_seq, output_seq, dirname, name):
    dirname = os.path.join('log', dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig, axes = plt.subplots(2, 4)
    cnt = 0
    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(array[cnt].transpose(-1, -2))
            axes[i, j].set_yticks(np.arange(len(input_seq)))
            axes[i, j].set_xticks(np.arange(len(output_seq)))
            axes[i, j].set_yticklabels(input_seq, fontsize=4)
            axes[i, j].set_xticklabels(output_seq, fontsize=4)
            axes[i, j].set_title('head_{}'.format(cnt), fontsize=10)
            plt.setp(axes[i, j].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            cnt += 1

    fig.suptitle(name, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, '{}.pdf'.format(name)))
    plt.close()

def draw_atts(maps, src, tgt, dirname, prefix):
    '''
    maps[0]: encoder self-attention
    maps[1]: encoder-decoder attention
    maps[2]: decoder self-attention
    '''
    draw_heatmap(maps[0], src, src, dirname, '{}_enc_self_attn'.format(prefix))
    draw_heatmap(maps[1], src, tgt, dirname, '{}_enc_dec_attn'.format(prefix))
    draw_heatmap(maps[2], tgt, tgt, dirname, '{}_dec_self_attn'.format(prefix))

mode2id = {'e2e': 0, 'e2d': 1, 'd2d': 2}

def att_animation(maps_array, mode, src, tgt, head_id):
    weights = [maps[mode2id[mode]][head_id] for maps in maps_array]
    fig, axes = plt.subplots(1, 2)
    axes[0].set_yticks(np.arange(len(src)))
    axes[0].set_xticks(np.arange(len(tgt)))
    axes[0].set_yticklabels(src)
    axes[0].set_xticklabels(tgt)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    def weight_animate(i):
        axes[0].cla()
        axes[0].set_title('heatmap')
        fig.suptitle('epoch {}'.format(i))
        weight = weights[i].transpose(-1, -2)
        heatmap = axes[0].pcolor(weight, vmin=0, vmax=1, cmap=plt.cm.Blues)
        plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_aspect('equal')
        axes[1].cla()
        axes[1].axis("off")
        graph_att_head(src, tgt, weight, axes[1], 'graph')

    ani = animation.FuncAnimation(fig, weight_animate, frames=len(weights), interval=500, repeat_delay=2000)
    return ani

def graph_att_head(M, N, weight, ax, title):
    "credit: Jinjing Zhou"
    in_nodes=len(M)
    out_nodes=len(N)

    g = nx.bipartite.generators.complete_bipartite_graph(in_nodes,out_nodes)
    X, Y = bipartite.sets(g)
    height_in = 10
    height_out = height_in 
    height_in_y = np.linspace(0, height_in, in_nodes)
    height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
    pos = dict()
    pos.update((n, (1, i)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
    pos.update((n, (3, i)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2
    ax.axis('off')
    ax.set_xlim(-1,4)
    ax.set_title(title)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='r', node_size=50, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='b', node_size=50, ax=ax)
    for edge in g.edges():
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=weight[edge[0], edge[1] - in_nodes] * 1.5, ax=ax)
    nx.draw_networkx_labels(g, pos, {i:label + '  ' for i,label in enumerate(M)},horizontalalignment='right', font_size=8, ax=ax)
    nx.draw_networkx_labels(g, pos, {i+in_nodes:'  ' + label for i,label in enumerate(N)},horizontalalignment='left', font_size=8, ax=ax)
