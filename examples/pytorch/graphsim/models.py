import dgl
from networkx.classes.reportviews import OutMultiEdgeView
import torch
import torch.nn as nn
from torch.nn import functional as F
import dgl.nn as dglnn
import dgl.function as fn
import copy

# Assume Global context has been merged to node feature initially
# Use Relu as default
class MLP(nn.Module):
    def __init__(self,in_feats,out_feats,num_layers):
        super(MLP,self).__init__()
        self.layers = nn.ModuleList()
        if num_layers > 1:
            for i in range(num_layers-1):
                self.layers.append(nn.Linear(in_feats,in_feats))
        self.layers.append(nn.Linear(in_feats,out_feats))

    # Input is expected to be: Size(n_particles,n_feats)
    # What if I want to do batch training ? Need to be considered
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class OnlinePrepareLayer(nn.Module):
    '''Prepare node feature and edge feature for computation
    It will be initialized with simulation parameters.

    Parameters
    ==========
    num_particles : int
        number of particles used in this simlation

    pos_dim : int
        dimension of position can be either 2 or 3-D

    hist_len : int
        length of historical horizon

    boundary_info : torch.Tensor
        length of

    Return
    ==========
    node_feature: Merged with global context feature if available
    edge_feature: Generated edge feature if consider the relative pose
    '''
    def __init__(self,
                 num_particles,
                 env_dim,
                 hist_len,
                 boundary_info,
                 particle_stats,
                 radius):
        super(OnlinePrepareLayer,self).__init__()
        self.num_particles = num_particles
        self.env_dim = env_dim
        self.hist_len = hist_len
        self.boundary_info = boundary_info # Tensor
        #{'front_bound':float,'back_bound':float,'left_bound':float,
        # 'right_bound':float,'top_bound':float,'bottom_bound':float}
        # Personally feels the boundary handling is bit wired
        self.p_stats = particle_stats
        self.radius = radius
        self.reset()

    def forward(self,g,pos):
        with g.local_scope():
            if self.init:
                for i in range(self.hist_len):
                    self.pos_hist[i] = copy.deepcopy(pos)
                    self.vel_hist[i] = torch.zeros(self.num_particles,
                                                self.env_dim)
                    self.normalized_vel_hist[i] = torch.zeros(self.num_particles,self.env_dim)
                self.init = False
                    
            else:
                self.pos_hist.pop()
                # Why normalize like this? How to preserve scale information of interactions?
                # Connetivity radius plays an important role.
                self.pos_hist.insert(0,pos)
                self.vel_hist.pop()
                self.vel_hist.insert(0,pos-self.pos_hist[1])
                self.normalized_vel_hist.pop()
                self.normalized_vel_hist.insert(0,(pos-self.pos_hist[1]-self.p_stats['vel_mean'])/self.p_stats['vel_std'])

            # Need to handle carefully

            pos2bound = pos.repeat_interleave(2,dim=1) - self.boundary_info.repeat(self.num_particles).view(self.num_particles,-1)
            normalized_pos2bound = torch.clamp(pos2bound/self.radius,-1,1)

            node_feature = torch.hstack(self.normalized_vel_hist+[normalized_pos2bound]) # [v_x1,v_y1,v_z1,v_x2,v_y2,v_z2,...,clip_norm_bounds]
            
            # Compute Edge data
            g.ndata['pos'] = pos
            g.apply_edges(fn.u_sub_v('pos','pos','rel_pos'))
            rel_pos = g.edata['rel_pos']/self.radius
            # May be just scaling issue since all the mean and variance are predefined. 
            rel_dis= torch.norm(rel_pos,dim=1,p=2).view(-1,1)
            edge_feature = torch.hstack([rel_pos,rel_dis])
            return node_feature.float(), edge_feature.float()

    def reset(self):
        self.pos_hist = [None for i in range(self.hist_len)]
        self.vel_hist = [None for i in range(self.hist_len)]
        self.normalized_vel_hist = [None for i in range(self.hist_len)]
        self.init = True
        # Need to handle boundary condition
        
class OfflinePrepareLayer(nn.Module):
    def __init__(self,
                 num_particles,
                 batch_size,
                 env_dim,
                 boundary_info,
                 particle_stats,
                 radius):
        super(OfflinePrepareLayer,self).__init__()
        self.num_particles = num_particles
        self.env_dim = env_dim
        self.boundary_info = boundary_info
        self.p_stats = particle_stats
        self.radius = radius
        self.batch_size = batch_size

    # Assume v is a [num_particles,dim*hist_length]
    def normalize_velocity(self,v):
        normalized_v = copy.deepcopy(v)
        normalized_v = (normalized_v-self.p_stats['vel_mean'])/self.p_stats['vel_std']
        return normalized_v
    
    # It can handle arbitrary baycjed graph
    def forward(self,g,pos,hist_v):
        with g.local_scope():
            current_v = hist_v[:,torch.arange(self.env_dim)] # Select last velocity
            normalized_v = self.normalize_velocity(hist_v)
            pos2bound = pos.repeat_interleave(2,dim=1) - self.boundary_info.repeat(pos.shape[0]).view(pos.shape[0],-1)
            normalized_pos2bound = torch.clamp(pos2bound/self.radius,-1,1)
            node_feature = torch.hstack([normalized_v,normalized_pos2bound])

            # Compute edge feature:
            g.ndata['pos'] = pos
            g.apply_edges(fn.u_sub_v('pos','pos','rel_pos'))
            rel_pos = g.edata['rel_pos']/self.radius
            rel_dis = torch.norm(rel_pos,dim=1,p=2).view(-1,1)
            edge_feature = torch.hstack([rel_pos,rel_dis])
            
            return node_feature.float(),edge_feature.float(),current_v.float()

class InteractionLayer(nn.Module):
    def __init__(self,in_node_feats,
                      in_edge_feats,
                      out_node_feats,
                      out_edge_feats):
        super(InteractionLayer,self).__init__()
        self.in_node_feats   = in_node_feats
        self.in_edge_feats   = in_edge_feats
        self.out_edge_feats  = out_edge_feats
        self.out_node_feats  = out_node_feats

        # MLP for message passing
        self.edge_fc = nn.Linear(2*self.in_node_feats+self.in_edge_feats,
                                 self.out_edge_feats)

        self.node_fc = nn.Linear(self.in_node_feats+self.out_edge_feats,
                                 self.out_node_feats)

    # Should be done by apply edge
    def update_edge_fn(self,edges):
        x = torch.cat([edges.src['feat'],edges.dst['feat'],edges.data['feat']],dim=1)
        return {'e':self.edge_fc(x)}

    # Assume agg comes from build in reduce
    def update_node_fn(self,nodes):
        x = torch.cat([nodes.data['feat'],nodes.data['agg']],dim=1)
        return {'n':self.node_fc(x)}

    def forward(self,g,node_feats,edge_feats):
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_feats
        g.apply_edges(self.update_edge_fn)
        g.update_all(fn.copy_e('e','msg'),
                     fn.mean('msg','agg'),
                     self.update_node_fn)
        return g.ndata['n'], g.edata['e']
    
class InputLayer(nn.Module):
    '''Element wisely process the data with MLP and layer norm
    Parameters
    ==========
    in_node_feats : int
        input node feature size
    
    in_edge_feats : int
        input edge feature size

    out_node_feats : int
        output node feature size

    out_edge_feats : int
        output edge feature size

    num_mlp_layers : int
        number of MLP layers

    layer norm : bool
        whether use layer norm or not

    global_context_feats : int default 0
        num of feature of global context default zero

    Returns
    ==========
    n_out : Tensor [n_particles,out_feats]
        output hidden feature

    e_out : Tensor [n_temp_edge,out_feats]
    '''
    def __init__(self,in_node_feats,
                      in_edge_feats,
                      out_node_feats,
                      out_edge_feats,
                      num_mlp_layers,
                      layer_norm=True,
                      global_context_feats=0):
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.out_node_feats= out_node_feats
        self.out_edge_feats= out_edge_feats
        self.global_context_feats = global_context_feats
        self.num_mlp_layers = num_mlp_layers
        self.layer_norm = layer_norm
        super(InputLayer,self).__init__()
        # Define nn
        self.fc_node = MLP(self.in_node_feats+self.global_context_feats,self.out_node_feats,self.num_mlp_layers)
        if layer_norm:
            self.ln_node = nn.LayerNorm(self.out_node_feats,elementwise_affine=True)

        self.fc_edge = MLP(self.in_edge_feats,self.out_edge_feats,self.num_mlp_layers)
        if layer_norm:
            self.ln_edge = nn.LayerNorm(self.out_edge_feats,elementwise_affine=True)

    # Assume global context has same dimension as node in the first dimension
    def forward(self,node_feature,edge_feature,global_context=None):
        if global_context!= None:
            node_feature = torch.hstack([node_feature,global_context])
        n_out = self.fc_node(node_feature)
        if self.layer_norm:
            n_out = self.ln_node(n_out)

        e_out = self.fc_edge(edge_feature)
        if self.layer_norm:
            e_out = self.ln_edge(e_out)

        return n_out, e_out

class OutputLayer(nn.Module):
    '''Only output node prediction
    Parameters
    ==========
    in_node_feats : int
        input node feature

    out_node_feats : int
        output node feature

    num_layers : int
        number of layers

    Forward Returns
    ==========
    pred : Tensor
        Prediction of next position or velocity
    '''
    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 num_layers):
        self.in_feats = in_node_feats
        self.out_feats= out_node_feats
        self.num_layers = num_layers
        super(OutputLayer,self).__init__()
        self.fc = MLP(self.in_feats,
                      self.out_feats,
                      self.num_layers)

    def forward(self,node_feature):
        out = self.fc(node_feature)
        return out

class InteractionGNN(nn.Module):
    def __init__(self,
                num_layers,
                in_node_feats,
                in_edge_feats,
                hid_node_feats,
                hid_edge_feats,
                out_node_feats,
                out_edge_feats,
                global_feats=0,
                num_mlp_layers=2,
                layer_norm = True,
                
                ):
        super(InteractionGNN,self).__init__()
        self.num_layers = num_layers
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.out_node_feats= out_node_feats
        self.out_edge_feats= out_edge_feats
        self.hid_node_feats= hid_node_feats
        self.hid_edge_feats= hid_edge_feats
        self.global_feats  = global_feats

        self.num_mlp_layers = num_mlp_layers
        self.layer_norm = layer_norm

        self.input_layer = InputLayer(self.in_node_feats,
                                      self.in_edge_feats,
                                      self.hid_node_feats,
                                      self.hid_edge_feats,
                                      self.num_mlp_layers,
                                      self.layer_norm,
                                      self.global_feats)


        self.interaction_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.interaction_layers.append(InteractionLayer(in_node_feats=self.hid_node_feats,
                                                            in_edge_feats=hid_edge_feats,
                                                            out_node_feats=self.hid_node_feats,
                                                            out_edge_feats=self.hid_edge_feats))

        self.output_layer = OutputLayer(self.hid_node_feats,
                                        self.out_node_feats,
                                        self.num_mlp_layers)

    def forward(self,g,node_feature,edge_feature,global_context=None):
        h_n,h_e = self.input_layer(node_feature,edge_feature,global_context)
        for layer in self.interaction_layers:
            n_out,e_out = layer(g,h_n,h_e)
            h_n = n_out + h_n
            h_e = e_out + h_e
        #print("After Input:",h_n,h_e)
        h_n = self.output_layer(h_n)
        return h_n

# Unit Test

if __name__ == "__main__":
    ignn = InteractionGNN(num_layers=10,
                          in_node_feats=3,
                          in_edge_feats=3,
                          hid_node_feats=5,
                          hid_edge_feats=5,
                          out_node_feats=5,
                          out_edge_feats=5,
                          global_feats=2,
                          num_mlp_layers=2)

    g = dgl.graph(([0,1,2,3,4,5],[1,2,3,4,5,0]))
    nf = torch.rand((6,3))+1
    ef = torch.rand((6,3))+1
    gc = torch.rand((6,2))+1
    out = ignn(g,nf,ef,gc)
    print(out)

    particle_stats = {'vel_mean':0.5,'vel_std':0.5}
    p_layer = OfflinePrepareLayer(6,3,torch.tensor([1,-1,1,-1,1,-1]).float(),particle_stats=particle_stats,radius=0.3)
    for i in range(10):
        nf = torch.rand((6,3))
        hist_v = torch.arange(15).repeat(6).view(6,-1)
        no,eo,cv = p_layer(g,nf,hist_v)
        print(no.shape,eo.shape,cv)