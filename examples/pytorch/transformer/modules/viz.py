import os
import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_attention_map(g, src_nodes, dst_nodes):
    """
    To visualize the attention score between two set of nodes.
    """
    n, m, h = len(src_nodes), len(dst_nodes), 8
    weight = th.zeros(n, m, h).fill_(-1e8)
    for i, src in enumerate(src_nodes.tolist()):
        for j, dst in enumerate(dst_nodes.tolist()):
            if not g.has_edge_between(src, dst):
                continue
            eid = g.edge_id(src, dst)
            weight[i][j] = g.edata['score'][eid].cpu().detach().view(-1)

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
            cnt += 1
            plt.setp(axes[i, j].get_xticklabels(), rotation=45, ha="right",
                             rotation_mode="anchor")
    fig.suptitle(name, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, '{}.pdf'.format(name)))
    plt.close()

def draw_atts(maps, src, tgt, dirname, prefix):
    draw_heatmap(maps[0], src, src, dirname, '{}_enc_self_attn'.format(prefix))
    draw_heatmap(maps[1], src, tgt, dirname, '{}_enc_dec_attn'.format(prefix))
    draw_heatmap(maps[2], tgt, tgt, dirname, '{}_dec_self_attn'.format(prefix))
