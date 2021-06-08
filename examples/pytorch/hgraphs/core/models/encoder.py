import dgl, torch
from core.models.utils import init_features_for_hgraph, embed

class Encoder(torch.nn.Module):
    def __init__(self, hidden_size, node_rep_size, latent_size,
                 dropout, embeddors, message_passing_net, device):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.node_rep_size = node_rep_size
        self.latent_size = latent_size
        self.latent_projector = LatentProjector(node_rep_size, latent_size, dropout)
        self.embeddors = embeddors
        self.message_passing_net = message_passing_net
        self.edge_types_to_embed = (
            ("atom", "bond", "atom"),
            ("attachment_config", "ancestry", "attachment_config"),
            ("motif", "ancestry", "motif")
        )

        self.device = device
    
    def forward(self, hgraphs):
        hgraphs_batched = dgl.batch(hgraphs)
        
        init_features_for_hgraph(hgraphs_batched, self.hidden_size, self.node_rep_size)
        for canonical_edge_type in self.edge_types_to_embed:
            node_type, edge_type, _ = canonical_edge_type
            embed(hgraphs_batched, self.embeddors, "nodes", node_type,
                  0, hgraphs_batched.number_of_nodes(node_type) - 1)
            embed(hgraphs_batched, self.embeddors, "edges", canonical_edge_type,
                  0, hgraphs_batched.number_of_edges(canonical_edge_type) - 1)
        
        self.message_passing_net(hgraphs_batched)
        hgraphs = dgl.unbatch(hgraphs_batched)
        
        root_motif_reps = torch.stack([hgraph.nodes["motif"].data["rep"][0] for hgraph in hgraphs])
        latents = self.latent_projector(root_motif_reps)

        return latents

class LatentProjector(torch.nn.Module):
    def __init__(self, node_rep_size, latent_size, dropout):
        """
        Projects a node representation to the latent space.
        """
        super().__init__()

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(node_rep_size, latent_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_size, latent_size)
        )
    
    def forward(self, node_rep):
        latent = self.MLP(node_rep)
        
        return latent
