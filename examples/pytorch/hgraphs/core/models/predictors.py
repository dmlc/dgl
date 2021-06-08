import torch

class NodePredictor(torch.nn.Module):
    def __init__(self, node_rep_size, latent_size, vocab_size, dropout):
        super().__init__()

        self.logit_MLP = torch.nn.Sequential(
            torch.nn.Linear(node_rep_size + latent_size, node_rep_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(node_rep_size, vocab_size)
        )
    
    def forward(self, frontier_node_reps, inputs):
        concat = torch.cat([frontier_node_reps, inputs], dim=1)
        logits = self.logit_MLP(concat)

        return logits

class AttachmentPredictor(torch.nn.Module):
    def __init__(self, node_rep_size, latent_size, dropout, method = "dot"):
        super().__init__()

        self.method = method
        self.node_rep_size = node_rep_size

        node_type_rep_size = node_id_rep_size = node_rep_size
        self.position_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_type_rep_size + node_id_rep_size, node_rep_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(node_rep_size, node_rep_size)
        )
        
        if self.method == "dot":
            self.logit_MLP = torch.nn.Sequential(
                torch.nn.Linear(node_rep_size + node_rep_size, node_rep_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(node_rep_size, latent_size)
            )
        elif self.method == "concat":
            self.logit_MLP = torch.nn.Sequential(
                torch.nn.Linear(node_rep_size + node_rep_size + latent_size, node_rep_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(node_rep_size, 1)
            )
        else:
            raise ValueError("Invalid method specified: must be either 'dot' or 'concat'.")

    def forward(self, atom_rep_pairs, inputs):
        """
        First get a tensor where, for batch element i and attachment pair j, (i, j, k, :)
        is the concat of the reps of the kth (i.e. 1st or 2nd) atom of the pair's type and
        node ID in the motif.
        Then pass (i, j, k, :) through the position_encoder MLP to get a position-informed
        representation of the atom in (i, j, k, :).
        
        Then, if method is "concat":
            Get a tensor where, for batch element i, (i, j, :) is the concat of its input and
        the position-informed representations of its jth pair of possible attachment atoms,
        and pass each (i, j, :) through an MLP to get logits for each pair j for each batch el i.
        Otherwise, if method is "dot":
            Get a tensor where, for batch element i, (i, j, :) is the concat of the position-informed
        representations of its jth pair of possible attachment atoms, and pass them through an MLP
        to get representations of the pairs. Then take the dot product between each of i's pairs'
        representations and the ith input to get input-conditioned logits for each pair j for
        each batch el i.

        Args:
            atom_rep_pairs - (graph i, attch pair j, atom k,
                [atom type rep, atom motif id rep], embed dim l)
            inputs - (graph i, input dim j)
        """

        atom_type_rep_id_rep_concat = atom_rep_pairs.reshape(
            atom_rep_pairs.shape[0], atom_rep_pairs.shape[1],
            atom_rep_pairs.shape[2], 2 * self.node_rep_size
        )
        pos_encoded_atom_rep_pairs = self.position_encoder(atom_type_rep_id_rep_concat)

        atom_rep_pairs_concat = pos_encoded_atom_rep_pairs.reshape(
            atom_rep_pairs.shape[0], atom_rep_pairs.shape[1], 2 * self.node_rep_size
        )
        inputs = inputs.unsqueeze(1)
        
        if self.method == "concat":
            inputs = inputs.expand(-1, atom_rep_pairs_concat.shape[1], -1)
            concat = torch.cat([atom_rep_pairs_concat, inputs], dim = 2)
            logits = self.logit_MLP(concat).squeeze(2)
        elif self.method == "dot":
            unconditional_logits = self.logit_MLP(atom_rep_pairs_concat)
            inputs = inputs.transpose(1,2)
            logits = torch.bmm(unconditional_logits, inputs).squeeze(2)

        return logits
