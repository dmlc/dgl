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
            eid = g.edge_ids(src, dst)
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

colorbar = None

def att_animation(maps_array, mode, src, tgt, head_id):
    weights = [maps[mode2id[mode]][head_id] for maps in maps_array]
    fig, axes = plt.subplots(1, 2)

    def weight_animate(i):
        global colorbar
        if colorbar:
            colorbar.remove()
        plt.cla()
        axes[0].set_title('heatmap')
        axes[0].set_yticks(np.arange(len(src)))
        axes[0].set_xticks(np.arange(len(tgt)))
        axes[0].set_yticklabels(src)
        axes[0].set_xticklabels(tgt)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fig.suptitle('epoch {}'.format(i))
        weight = weights[i].transpose(-1, -2)
        heatmap = axes[0].pcolor(weight, vmin=0, vmax=1, cmap=plt.cm.Blues)
        colorbar = plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_aspect('equal')
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
    
    
    
import networkx as nx
from networkx.utils import is_string_like
from matplotlib.patches import ConnectionStyle,FancyArrowPatch
"The following function was modified from the source code of networkx"
def draw_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=1.0,
                        arrowstyle='-|>',
                        arrowsize=10,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        label=None,
                        node_size=300,
                        nodelist=None,
                        node_shape="o",
                        connectionstyle='arc3',
                        **kwds):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=1.0)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    label : [None| string]
       Label for legend

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size' as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch, ConnectionStyle
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if nodelist is None:
        nodelist = list(G.nodes())

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not is_string_like(edge_color) \
            and cb.iterable(edge_color) \
            and len(edge_color) == len(edge_pos):
        if np.alltrue([is_string_like(c) for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif np.alltrue([not is_string_like(c) for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4)
                           for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must contain color names or numbers')
    else:
        if is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            msg = 'edge_color must be a color or list of one color per edge'
            raise ValueError(msg)

    if (not G.is_directed() or not arrows):
        edge_collection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=lw,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        # Note: there was a bug in mpl regarding the handling of alpha values
        # for each line in a LineCollection. It was fixed in matplotlib by
        # r7184 and r7189 (June 6 2009). We should then not set the alpha
        # value globally, since the user can instead provide per-edge alphas
        # now.  Only set it globally if provided as a scalar.
        if cb.is_numlike(alpha):
            edge_collection.set_alpha(alpha)

        if edge_colors is None:
            if edge_cmap is not None:
                assert(isinstance(edge_cmap, Colormap))
            edge_collection.set_array(np.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
            if edge_vmin is not None or edge_vmax is not None:
                edge_collection.set_clim(edge_vmin, edge_vmax)
            else:
                edge_collection.autoscale()
        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head
        arrow_colors = edge_colors
        if arrow_colors is None:
            if edge_cmap is not None:
                assert(isinstance(edge_cmap, Colormap))
            else:
                edge_cmap = plt.get_cmap()  # default matplotlib colormap
            if edge_vmin is None:
                edge_vmin = min(edge_color)
            if edge_vmax is None:
                edge_vmax = max(edge_color)
            color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)

        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            arrow_color = None
            line_width = None
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if cb.iterable(node_size):  # many node sizes
                src_node, dst_node = edgelist[i]
                index_node = nodelist.index(dst_node)
                marker_size = node_size[index_node]
                shrink_target = to_marker_edge(marker_size, node_shape)
            else:
                shrink_target = to_marker_edge(node_size, node_shape)
            if arrow_colors is None:
                arrow_color = edge_cmap(color_normal(edge_color[i]))
            elif len(arrow_colors) > 1:
                arrow_color = arrow_colors[i]
            else:
                arrow_color = arrow_colors[0]
            if len(lw) > 1:
                line_width = lw[i]
            else:
                line_width = lw[0]
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle=arrowstyle,
                                    shrinkA=shrink_source,
                                    shrinkB=shrink_target,
                                    mutation_scale=mutation_scale,
                                    connectionstyle=connectionstyle,
                                    color=arrow_color,
                                    linewidth=line_width,
                                    zorder=1)  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx,  pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return arrow_collection


def draw_g(graph):
    g=graph.g.to_networkx()
    fig=plt.figure(figsize=(8,4),dpi=150)
    ax=fig.subplots()
    ax.axis('off')
    ax.set_ylim(-1,1.5)
    en_indx=graph.nids['enc'].tolist()
    de_indx=graph.nids['dec'].tolist()
    en_l={i:np.array([i,0]) for i in en_indx}
    de_l={i:np.array([i+2,1]) for i in de_indx}
    en_de_s=[]
    for i in en_indx:
        for j in de_indx:
            en_de_s.append((i,j))
            g.add_edge(i,j)
    en_s=[]
    for i in en_indx:
        for j in en_indx:
            g.add_edge(i,j)
            en_s.append((i,j))

    de_s=[]
    for idx,i in enumerate(de_indx):
        for j in de_indx[idx:]:
            g.add_edge(i,j)
            de_s.append((i,j))


    nx.draw_networkx_nodes(g, en_l, nodelist=en_indx, node_color='r', node_size=60, ax=ax)
    nx.draw_networkx_nodes(g, de_l, nodelist=de_indx, node_color='r', node_size=60, ax=ax)
    draw_networkx_edges(g,en_l,edgelist=en_s, ax=ax,connectionstyle="arc3,rad=-0.3",width=0.5)
    draw_networkx_edges(g,de_l,edgelist=de_s, ax=ax,connectionstyle="arc3,rad=-0.3",width=0.5)
    draw_networkx_edges(g,{**en_l,**de_l},edgelist=en_de_s,width=0.3, ax=ax)
    # ax.add_patch()
    ax.text(len(en_indx)+0.5,0,"Encoder", verticalalignment='center', horizontalalignment='left')

    ax.text(len(en_indx)+0.5,1,"Decoder", verticalalignment='center', horizontalalignment='right')
    delta=0.03
    for value in {**en_l,**de_l}.values():
        x,y=value
        ax.add_patch(FancyArrowPatch((x-delta,y+delta),(x-delta,y-delta),arrowstyle="->",mutation_scale=8,connectionstyle="arc3,rad=3"))
    plt.show(fig)
