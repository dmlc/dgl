import copy, dgl, torch
from core.models.utils import get_empty_hgraph, init_features_for_hgraph, add_graph_to_graph, embed
from core.models.predictors import NodePredictor, AttachmentPredictor

from collections import namedtuple
FrontierNodeIds = namedtuple("frontier_node_ids", ["motif", "attch_conf"])

class Decoder(torch.nn.Module):
    def __init__(self, vocabs, motif_graphs, hidden_size, node_rep_size,
                 latent_size, attach_prediction_method, dropout,
                 embeddors, message_passing_net, device):
        super().__init__()
        
        self.vocabs = vocabs
        self.motif_graphs = motif_graphs
        self.hidden_size = hidden_size
        self.node_rep_size = node_rep_size

        self.motif_predictor = NodePredictor(
            node_rep_size, latent_size,
            len(vocabs["motif"]["node"]), dropout
        )
        self.attch_conf_predictor = NodePredictor(
            node_rep_size, latent_size,
            vocabs["attachment_config"]["max_per_motif"], dropout
        )
        self.attch_predictor = AttachmentPredictor(node_rep_size, latent_size, dropout, attach_prediction_method)

        self.loss_func = torch.nn.NLLLoss(reduction = "none")

        self.embeddors = embeddors
        self.message_passing_net = message_passing_net

        self.device = device
        self.to(device)
  
    def forward(self, inputs, target_graphs = None, max_motifs = None):
        """
        Args:
        inputs - Matrix whose rows have dimension latent_size.
        target_graphs - List of graphs to train model to generate for each input.
        max_motifs - Max num motifs to generate before stopping.
        """

        ctx = {
            "input": inputs,
            "pred_hgraph": [],
            "frontier_node_ids": [], #[b][i] := ith FrontierNodeIds instance in the bth graph's frontier.
            "motif_to_mol_atom_ids": [], #[b][i][j] := id of motif i's jth atom in the bth graph.
            "avail_attchs": [], #[b][i][j] := ids of motif i's unused attachment atoms of type j in its motif graph.
            "ancestors": [] #[b][i] := ids of motif i's ancestors in the bth graph.
        }
        for _ in range(inputs.shape[0]):
            ctx["pred_hgraph"].append(get_empty_hgraph(self.hidden_size, self.node_rep_size, device = self.device))
            ctx["frontier_node_ids"].append([])
            ctx["motif_to_mol_atom_ids"].append([])
            ctx["avail_attchs"].append([])
            ctx["ancestors"].append([])

        #Map a graph's idx in the sub-batch being predicted to its idx in the whole batch of graphs,
        #so after the bth graph is done being predicted, it can be removed and skipped over.
        ctx["pred_to_batch_idx"] = list(range(inputs.shape[0]))
        
        if target_graphs is None:
            hgraphs = self._infer(ctx, max_motifs)
            return hgraphs
        else:
            hgraphs, loss = self._train(ctx, target_graphs)
            return hgraphs, loss

    def _train(self, ctx, target_graphs):
        """
        Traverse the target graphs depth-first and record how far the model
        is from the correct motif/attachment_config/attachment at
        each step.
        """

        losses = []
        
        """
        Stacks of the motifs that still need to be added and have their children traversed for each graph in the batch.
        stacks[b][i] := (
            motif node ID OR "stop",
            (parent attachment atom, child attachment atom) OR (-1, -1) when "stop")
        )
        """
        stacks = [[(0, torch.tensor([-1,-1]))] for _ in range(ctx["input"].shape[0])]
        
        targets = { "motif_idx": torch.zeros(len(target_graphs), dtype = torch.long, device = self.device),
                    "attch_conf_idx": torch.zeros(len(target_graphs), dtype = torch.long, device = self.device),
                    "attch_pair": torch.zeros(len(target_graphs), 2, dtype = torch.long, device = self.device) }
        
        #Loop until there are no graphs left to predict in the batch.
        while len(ctx["pred_to_batch_idx"]) != 0:
            remaining_batch_idxs = []
            for batch_idx, stack in enumerate(stacks):
                if len(stack) > 0:
                    self.proceed_through_stack(
                        ctx, stack, batch_idx, remaining_batch_idxs,
                        targets, target_graphs)
                else:
                    targets["motif_idx"][batch_idx] = -1
                    targets["attch_conf_idx"][batch_idx] = -1
                    targets["attch_pair"][batch_idx] = torch.tensor([-1, -1])

            ctx["pred_to_batch_idx"][:] = remaining_batch_idxs 
            if len(ctx["pred_to_batch_idx"]) > 0:
                pred_hgraphs = self.decode_step(ctx, targets, losses)
        
        loss = torch.mean(torch.stack(losses))
       
        return ctx["pred_hgraph"], loss

    def proceed_through_stack(self, ctx, stack, batch_idx, remaining_batch_idxs,
                              targets, target_graphs):
        """
        Pop "stop" signals off the top of the stack until it's either empty,
        or there's a motif idx at the top, in which case make it the target
        of the decoder's next step.
        """

        while len(stack) > 0:
            target_motif_idx, target_attch_conf_idx, target_attch_pair = (
                self.next_node(stack, target_graphs[batch_idx])
            )
           
            if target_motif_idx == "stop":
                del ctx["frontier_node_ids"][batch_idx][-1]
            else:
                targets["motif_idx"][batch_idx] = target_motif_idx
                targets["attch_conf_idx"][batch_idx] = target_attch_conf_idx
                targets["attch_pair"][batch_idx] = target_attch_pair
                remaining_batch_idxs.append(batch_idx)
                break
    
        #The last element in the stack was "stop" signal.
        if target_motif_idx == "stop":
            targets["motif_idx"][batch_idx] = -1
            targets["attch_conf_idx"][batch_idx] = -1
            targets["attch_pair"][batch_idx] = torch.tensor([-1, -1])

        
    def next_node(self, stack, target_graph):
        node_id, target_attch_pair = stack.pop()
        target_motif_idx, target_attch_conf_idx = self.get_next_targets(node_id, target_graph)
        if target_motif_idx != "stop":
            self.add_children_to_stack(node_id, stack, target_graph)

        return target_motif_idx, target_attch_conf_idx, target_attch_pair
    
    def get_next_targets(self, node_id, target_graph):
        if node_id == "stop":
            target_motif_idx = "stop"
            target_attch_conf_idx = None
        else:
            target_motif_idx = (
                target_graph.nodes["motif"]
                            .data["vocab_idx"][node_id]
            )
            target_attch_conf_idx = (
                target_graph.nodes["attachment_config"]
                            .data["vocab_idx"][node_id]
            )

        return target_motif_idx, target_attch_conf_idx

    def add_children_to_stack(self, node_id, stack, target_graph):
            parent_ids, child_ids, edge_ids = target_graph.out_edges(
                node_id, etype = ("motif", "attaches to", "motif"), form = "all"
            )
            
            #Stop once all a motif's children have been
            #popped off the frontier list (i.e. they and their descendents have
            #all been generated).
            stack.append(("stop", torch.tensor([-1, -1])))
            if len(child_ids) > 0:
                attachment_motif_id_pairs = (
                    target_graph.edges[("motif", "attaches to", "motif")]
                    .data["attachment_motif_id_pair"][edge_ids]
                )
                stack.extend(zip(child_ids.tolist(), attachment_motif_id_pairs))

    def _infer(self, ctx, max_motifs):
        """
        Generate hierarchical graphs representing molecules,
        conditioned on input vectors.
        """

        pred_hgraphs = self.decode_step(ctx)

        #Loop until there are no graphs left in the batch to predict.
        while len(ctx["pred_to_batch_idx"]) != 0:
            remaining_batch_idxs = []
            for pred_idx, batch_idx in enumerate(ctx["pred_to_batch_idx"]):
                frontier_isnt_empty = len(ctx["frontier_node_ids"][batch_idx]) > 0
                motifs_under_max = ctx["pred_hgraph"][batch_idx].number_of_nodes("motif") <= max_motifs
                if frontier_isnt_empty and motifs_under_max:
                    remaining_batch_idxs.append(batch_idx)
                    frontier_motif_id = ctx["frontier_node_ids"][batch_idx][-1].motif
                    no_avail_attchs = len(ctx["avail_attchs"][batch_idx][frontier_motif_id]) == 0
                    if no_avail_attchs:
                        del ctx["frontier_node_ids"][batch_idx][-1]
                        if len(ctx["frontier_node_ids"][batch_idx]) == 0:
                            del remaining_batch_idxs[-1]
            
            ctx["pred_to_batch_idx"][:] = remaining_batch_idxs
            if len(ctx["pred_to_batch_idx"]) > 0:
                ctx["pred_hgraph"][:] = self.decode_step(ctx)

        return ctx["pred_hgraph"]

    def decode_step(self, ctx, targets = None, losses = None):
        """
        Either predict the first motif and its attachment config,
        or predict a new motif to attach to the latest frontier motif,
        along with its attachment config and the specific pair of atoms
        to attach at, for a batch of inputs.

        If targets are specified, the predictions will be used only to record
        the loss of the model and will be overridden with the target values.
        """

        #Update all the embeddings to account for the new connections.
        pred_hgraphs_batched = dgl.batch(ctx["pred_hgraph"])
        self.message_passing_net(pred_hgraphs_batched)
        ctx["pred_hgraph"] = dgl.unbatch(pred_hgraphs_batched)

        #Context of this step, as opposed to broader context of the decoding process (ctx).
        step_ctx = {}

        #Based on stop signals and attachment availability,
        #some graphs may get excluded from the predictions temporarily for this step.
        step_ctx["pred_to_batch_idx"] = ctx["pred_to_batch_idx"].copy()
        step_ctx["frontier_motif_rep"] = self.get_frontier_motif_reps(ctx, step_ctx)
        step_ctx["new_motif_vocab_idx"] = self.get_new_motif_nodes(ctx, step_ctx, targets, losses)
        step_ctx["new_attch_conf_vocab_idx"], step_ctx["new_attch_conf"] = self.get_new_attch_conf_nodes(
            ctx, step_ctx, targets, losses
        )

        #See if it's possible to add & attach the new motif and attch conf,
        #and get info required to make the atom attachment.
        p_left = len(step_ctx["pred_to_batch_idx"])
        step_ctx["common_attch_types"] = [None] * p_left
        step_ctx["n_valid_attch_pairs_per_type"] = [None] * p_left
        step_ctx["valid_attch_motif_id_pairs"] = [None] * p_left
        step_ctx["valid_attch_mol_id_pairs"] = [None] * p_left
        step_ctx["valid_attch_rep_pairs"] = [None] * p_left
        step_ctx["valid_attch_motif_id_embed_pairs"] = [None] * p_left
        step_ctx["new_attch_conf_node_id"] = [None] * p_left
        step_ctx["new_motif_node_id"] = [None] * p_left
        max_attch_pairs = 0
        remaining_preds = []
        for pred_idx, batch_idx in enumerate(step_ctx["pred_to_batch_idx"]):
            stop = self.try_add_new_motif(batch_idx, ctx, pred_idx, step_ctx)
            if not stop:
                remaining_preds.append(pred_idx)
                max_attch_pairs = max(max_attch_pairs, len(step_ctx["valid_attch_rep_pairs"][pred_idx]))
        if len(remaining_preds) == 0:
            return ctx["pred_hgraph"]
        else:
            for key in step_ctx:
                step_ctx[key][:] = [step_ctx[key][p_idx] for p_idx in remaining_preds]
        step_ctx["max_attch_pairs"] = max_attch_pairs

        #Choose a pair of atoms to attach at and make the attachment.
        step_ctx["chosen_pair_idx"] = self.get_new_atom_attachments(ctx, step_ctx, targets, losses)
        for pred_idx, batch_idx in enumerate(step_ctx["pred_to_batch_idx"]):
            self.attach_new_atom_node(batch_idx, ctx, pred_idx, step_ctx)
            ctx["frontier_node_ids"][batch_idx].append(
                FrontierNodeIds(motif = step_ctx["new_motif_node_id"][pred_idx],
                                attch_conf = step_ctx["new_attch_conf_node_id"][pred_idx])
            )
 
        return ctx["pred_hgraph"]

    def get_frontier_motif_reps(self, ctx, step_ctx):
        frontier_motif_reps = []

        for batch_idx, (pred_hgraph, frontier_node_ids) in enumerate(zip(ctx["pred_hgraph"], ctx["frontier_node_ids"])):
            if batch_idx in step_ctx["pred_to_batch_idx"]:
                if len(frontier_node_ids) != 0:
                    frontier_motif_rep = (pred_hgraph
                                          .nodes["motif"]
                                          .data["rep"]
                                          [frontier_node_ids[-1].motif])
                else:
                    #It's the initializer -- doesn't correspond to a real motif node.
                    frontier_motif_rep = torch.zeros(self.node_rep_size, device = self.device)
                frontier_motif_reps.append(frontier_motif_rep)

        return frontier_motif_reps

    def get_new_motif_nodes(self, ctx, step_ctx, targets, losses):
        inputs = ctx["input"][step_ctx["pred_to_batch_idx"]]
        frontier_motif_rep_tensor = torch.stack(step_ctx["frontier_motif_rep"])
        motif_logits = self.motif_predictor(frontier_motif_rep_tensor, inputs)
        
        if targets is not None:
            motif_log_probs = torch.nn.functional.log_softmax(motif_logits, dim = 1)
            target_motif_idxs = targets["motif_idx"][step_ctx["pred_to_batch_idx"]]
            pred_losses = self.loss_func(motif_log_probs, target_motif_idxs)
            losses.extend(pred_losses)
            new_motif_vocab_idxs = list(target_motif_idxs)
            sampled = list(torch.distributions.Categorical(logits = motif_logits).sample())
        else:
            #Categorical softmaxes logits by default.
            new_motif_vocab_idxs = list(torch.distributions.Categorical(logits = motif_logits).sample())
            sampled = new_motif_vocab_idxs
    
        return new_motif_vocab_idxs

    def get_new_attch_conf_nodes(self, ctx, step_ctx, targets, losses):
        inputs = ctx["input"][step_ctx["pred_to_batch_idx"]]
        frontier_motif_rep_tensor = torch.stack(step_ctx["frontier_motif_rep"])
        attch_confs_logits = self.attch_conf_predictor(frontier_motif_rep_tensor, inputs)
        if targets is not None:
            attch_confs_log_probs = torch.nn.functional.log_softmax(attch_confs_logits, dim = 1)
            target_attch_conf_idxs = targets["attch_conf_idx"][step_ctx["pred_to_batch_idx"]]
            pred_losses = self.loss_func(attch_confs_log_probs, target_attch_conf_idxs)
            losses.extend(pred_losses)
            new_attch_conf_vocab_idxs = list(target_attch_conf_idxs)
        else:
            for pred_idx, motif_vocab_idx in enumerate(step_ctx["new_motif_vocab_idx"]):
                num_valid_logits = len(self.vocabs["attachment_config"]["node"][motif_vocab_idx.item()])
                attch_confs_logits[pred_idx][num_valid_logits:] = float("-inf")
            attch_confs_probs = torch.softmax(attch_confs_logits, dim = 1)
            new_attch_conf_vocab_idxs = list(torch.distributions.Categorical(attch_confs_probs).sample())
    
        attch_confs = [
            copy.deepcopy(self.vocabs["attachment_config"]
                                     ["node"]
                                     [motif_vocab_idx.item()]
                                     [new_attch_conf_vocab_idx.item()])
            for motif_vocab_idx, new_attch_conf_vocab_idx
            in zip(step_ctx["new_motif_vocab_idx"], new_attch_conf_vocab_idxs)
        ]
        
        return new_attch_conf_vocab_idxs, attch_confs

    def try_add_new_motif(self, b, ctx, p, step_ctx):
        """
        Try to add the motif node, attachment config node, and atom graph of a new motif to the
        hitherto predicted hgraph, attach the motif and attachment config nodes to the corresponding
        frontier nodes, and return the possible attachments to make at the atom level.

        Args:
        b - The idx in the batch of the graph to which the motif is being added.
        p - The idx in the step's predictions of the graph to which the motif is being added.
        """

        is_first_motif = len(ctx["frontier_node_ids"][b]) == 0
        if not is_first_motif:
            #Find valid attchs between the new motif and frontier motif.
            frontier_motif_avail_attchs = ctx["avail_attchs"][b][ctx["frontier_node_ids"][b][-1].attch_conf]
            new_motif_avail_attchs = step_ctx["new_attch_conf"][p]
            step_ctx["common_attch_types"][p] = self.get_common_attch_types(frontier_motif_avail_attchs, new_motif_avail_attchs)
        else:
            step_ctx["common_attch_types"][p] = []

        can_attch_to_frontier_motif = len(step_ctx["common_attch_types"][p]) > 0
        if is_first_motif or can_attch_to_frontier_motif:
            self.add_new_motif(b, ctx, p, step_ctx)
            if can_attch_to_frontier_motif:
                self.attach_new_motif(b, ctx, p, step_ctx)
                stop = False
            else:
                #It's the first motif, so it can't attach to anything, but add it to the
                #frontier before stopping it.
                ctx["frontier_node_ids"][b].append(
                    FrontierNodeIds(motif = step_ctx["new_motif_node_id"][p],
                                    attch_conf = step_ctx["new_attch_conf_node_id"][p])
                ) 
          
                stop = True
        else:
            #If the chosen new motif isn't the initial motif and it can't attach to the frontier motif,
            #take that as a stop signal so it doesn't futiley keep trying to add things to the frontier motif.
            #Or should I give it a limited number of tries?
          
            del ctx["frontier_node_ids"][b][-1]
            stop = True
        
        return stop

    def add_new_motif(self, b, ctx, p, step_ctx):
        new_motif_node_id = self.add_new_motif_node(b, ctx, p, step_ctx)
        step_ctx["new_motif_node_id"][p] = new_motif_node_id

        new_attch_conf_node_id = self.add_new_attch_conf_node(
            b, ctx, p, step_ctx
        )
        step_ctx["new_attch_conf_node_id"][p] = new_attch_conf_node_id
        
        self.add_new_atoms(b, ctx, p, step_ctx)

    def attach_new_motif(self, b, ctx, p, step_ctx):
        """
        Attach the new motif and attachment config nodes to the frontier nodes
        and get the necessary information to find atoms to attach at.
        """
        
        ancestry_edges = self.get_ancestry_edges(
            ctx["ancestors"][b], step_ctx["new_motif_node_id"][p], ctx["frontier_node_ids"][b][-1].motif
        ) 
        self.attach_new_motif_node(
            ctx["pred_hgraph"][b], step_ctx["new_motif_node_id"][p], ctx["frontier_node_ids"][b][-1].motif, ancestry_edges
        )
        self.attach_new_attch_conf_node(
            ctx["pred_hgraph"][b], step_ctx["new_attch_conf_node_id"][p],
            ctx["frontier_node_ids"][b][-1].attch_conf, ancestry_edges
        )
        
        #We'll need the reps of the attachment atoms in each pair,
        #their ids local to their motifs, and their ids in the entire hier graph.
        (num_valid_attch_id_pairs_per_type, valid_attch_motif_id_pairs,
         valid_attch_mol_id_pairs, valid_attch_rep_pairs,
         valid_attch_motif_id_embed_pairs) = self.get_avail_attch_pairs(
            ctx["pred_hgraph"][b], step_ctx["common_attch_types"][p],
            ctx["avail_attchs"][b][ctx["frontier_node_ids"][b][-1].motif],
            ctx["avail_attchs"][b][step_ctx["new_motif_node_id"][p]],
            ctx["frontier_node_ids"][b][-1].motif, step_ctx["new_motif_node_id"][p],
            ctx["motif_to_mol_atom_ids"][b]
        )
        step_ctx["n_valid_attch_pairs_per_type"][p] = num_valid_attch_id_pairs_per_type
        step_ctx["valid_attch_motif_id_pairs"][p] = valid_attch_motif_id_pairs
        step_ctx["valid_attch_mol_id_pairs"][p] = valid_attch_mol_id_pairs
        step_ctx["valid_attch_rep_pairs"][p] = valid_attch_rep_pairs
        step_ctx["valid_attch_motif_id_embed_pairs"][p] = valid_attch_motif_id_embed_pairs

    def get_common_attch_types(self, frontier_attch_conf, new_attch_conf):
        """
        Returns:
        common_attch_types - List of the possible types (i.e. atomic numbers)
            of attachments between the motifs.
        """

        frontier_motif_attch_types = set(frontier_attch_conf.keys())
        new_motif_attch_types = set(new_attch_conf.keys())
        common_attch_types = (frontier_motif_attch_types.intersection(new_motif_attch_types))

        return common_attch_types 
    
    def add_new_motif_node(self, b, ctx, p, step_ctx):
        ctx["pred_hgraph"][b].add_nodes(
            1,
            {"vocab_idx": step_ctx["new_motif_vocab_idx"][p].unsqueeze(0)},
            ntype = "motif"
        )
        
        new_motif_node_id = ctx["pred_hgraph"][b].number_of_nodes("motif") - 1
        embed(ctx["pred_hgraph"][b], self.embeddors, "nodes", "motif",
              new_motif_node_id, new_motif_node_id)
        
        ctx["ancestors"][b].append([])
        return new_motif_node_id
    
    def add_new_attch_conf_node(self, b, ctx, p, step_ctx):
        ctx["pred_hgraph"][b].add_nodes(
            1, 
            {"vocab_idx": step_ctx["new_attch_conf_vocab_idx"][p].unsqueeze(0)},
            ntype = "attachment_config"
        )
        new_attch_conf_node_id = ctx["pred_hgraph"][b].number_of_nodes("attachment_config") - 1
        embed(ctx["pred_hgraph"][b], self.embeddors, "nodes",
              "attachment_config", new_attch_conf_node_id,
              new_attch_conf_node_id)

        ctx["avail_attchs"][b].append(step_ctx["new_attch_conf"][p])

        #Connect attch config to motif.
        ctx["pred_hgraph"][b].add_edges([new_attch_conf_node_id],
                                        [step_ctx["new_motif_node_id"][p]],
                                        etype = ("attachment_config", "of", "motif"))

        return new_attch_conf_node_id
    
    def add_new_atoms(self, b, ctx, p, step_ctx):
        new_motif_atom_graph = self.motif_graphs[step_ctx["new_motif_vocab_idx"][p]].to(self.device)
        init_features_for_hgraph(new_motif_atom_graph, self.hidden_size, self.node_rep_size)
        embed(new_motif_atom_graph, self.embeddors, "nodes",
              "atom", 0, new_motif_atom_graph.number_of_nodes("atom") - 1)
        embed(new_motif_atom_graph, self.embeddors, "edges",
              ("atom", "bond", "atom"), 0, new_motif_atom_graph.number_of_edges(("atom", "bond", "atom")) - 1)
        inserted_atoms_start_id = ctx["pred_hgraph"][b].number_of_nodes("atom")
        add_graph_to_graph(ctx["pred_hgraph"][b], new_motif_atom_graph)
        inserted_atoms_end_id = ctx["pred_hgraph"][b].number_of_nodes("atom") - 1 
        new_motif_atom_ids = torch.arange(inserted_atoms_start_id, inserted_atoms_end_id + 1, device = self.device)
        ctx["motif_to_mol_atom_ids"][b].append(new_motif_atom_ids)

        #Connect new atoms to the attch config.
        source_node_ids = torch.arange(inserted_atoms_start_id, inserted_atoms_end_id + 1, device = self.device)
        target_node_id = step_ctx["new_attch_conf_node_id"][p]
        ctx["pred_hgraph"][b].add_edges(source_node_ids, target_node_id, etype = ("atom", "of", "attachment_config"))
 
    def get_ancestry_edges(self, ancestors, new_motif_node_id, frontier_motif_node_id):
        """
        Record the new motif's ancestors.
        Add edges from the new motif to its ancestors with a feature specifying their
        distance, and edges from the new motif's ancestors to itself with a constant
        feature of 0.
        """
        
        ancestors[new_motif_node_id].append(frontier_motif_node_id)
        ancestors[new_motif_node_id].extend(ancestors[frontier_motif_node_id])
        
        num_ancestors = len(ancestors[new_motif_node_id])
        src_node_ids = []
        dst_node_ids = []
        ancestry_labels = []

        #New motif -> Ancestors
        src_node_ids.extend([new_motif_node_id] * num_ancestors)
        dst_node_ids.extend(ancestors[new_motif_node_id])

        #The Embeddor is only defined to embed upto the max depth of ancestors in the vocab,
        #so clip the depth labels of these edges to be at most the max of the vocab.
        max_ancestry_label = self.vocabs["motif"]["edge"][-1]
        if num_ancestors <= max_ancestry_label:
            clipped_ancestry_labels = range(1, num_ancestors + 1)
        else:
            clipped_ancestry_labels = list(range(1, max_ancestry_label + 1))
            remaining_ancestors = num_ancestors - max_ancestry_label
            clipped_ancestry_labels.extend([max_ancestry_label] * remaining_ancestors)
        ancestry_labels.extend(clipped_ancestry_labels)
        
        #Ancestors -> new motif
        src_node_ids.extend(ancestors[new_motif_node_id])
        dst_node_ids.extend([new_motif_node_id] * num_ancestors)
        ancestry_labels.extend([0] * num_ancestors)

        data = { "vocab_idx": torch.tensor(ancestry_labels, device = self.device) }
        edges = {"src": src_node_ids, "dst": dst_node_ids, "data": data}
        
        return edges

    def attach_new_motif_node(self, pred_hgraph, new_motif_node_id,
                              frontier_motif_node_id, ancestry_edges):
        #Add attachment edge from frontier motif to new motif. 
        pred_hgraph.add_edges(
            [frontier_motif_node_id],
            [new_motif_node_id], 
            etype = ("motif", "attaches to", "motif")
        )
        new_motif_edge_id = pred_hgraph.number_of_edges(
            ("motif", "attaches to", "motif")
        ) - 1

        #Add ancestry edges to/from all ctx["ancestors"].
        inserted_edges_start_id = pred_hgraph.number_of_edges(
            ("motif", "attaches to", "motif")
        )

        pred_hgraph.add_edges(
            ancestry_edges["src"],
            ancestry_edges["dst"],
            ancestry_edges["data"],
            etype = ("motif", "ancestry", "motif")
        )
        inserted_edges_end_id = pred_hgraph.number_of_edges(
            ("motif", "ancestry", "motif")
        ) - 1

        embed(pred_hgraph, self.embeddors, "edges",
              ("motif", "ancestry", "motif"),
              inserted_edges_start_id, inserted_edges_end_id)

    def attach_new_attch_conf_node(self, pred_hgraph, new_attch_conf_node_id,
                                   frontier_attch_conf_node_id, ancestry_edges):
        #Add attch edge from frontier attch conf to new attch conf. 
        pred_hgraph.add_edges(
            [frontier_attch_conf_node_id],
            [new_attch_conf_node_id],
            etype = ("attachment_config", "attaches to", "attachment_config")
        )
        new_attch_conf_edge_id = pred_hgraph.number_of_edges(
            ("attachment_config", "attaches to", "attachment_config")
        ) - 1

        #Add ancestry edges to all ancestors.
        inserted_edges_start_id = pred_hgraph.number_of_edges(
            ("attachment_config", "ancestry", "attachment_config")
        )
        pred_hgraph.add_edges(
            ancestry_edges["src"],
            ancestry_edges["dst"],
            ancestry_edges["data"],
            etype = ("attachment_config", "ancestry", "attachment_config")
        )
        inserted_edges_end_id = pred_hgraph.number_of_edges(
            ("attachment_config", "ancestry", "attachment_config")
        ) - 1
        
        embed(pred_hgraph, self.embeddors, "edges",
              ("attachment_config", "ancestry", "attachment_config"),
              inserted_edges_start_id, inserted_edges_end_id)

    def get_avail_attch_pairs(self, pred_hgraph, common_attch_types, frontier_motif_avail_attchs,
                              new_motif_avail_attchs, frontier_motif_node_id,
                              new_motif_node_id, motif_to_mol_atom_ids):
        num_valid_attch_id_pairs_per_type = []
        valid_attch_motif_id_pairs = torch.tensor([], dtype = torch.long, device = self.device)
        valid_attch_mol_id_pairs = torch.tensor([], dtype = torch.long, device = self.device)
        for type_ in common_attch_types:
            #IDs of the available attachment atoms in the motifs.
            attch_frontier_motif_ids = torch.tensor(frontier_motif_avail_attchs[type_], device = self.device)
            attch_new_motif_ids = torch.tensor(new_motif_avail_attchs[type_], device = self.device)
            typed_attch_motif_id_pairs = torch.cartesian_prod(attch_frontier_motif_ids, attch_new_motif_ids)
            valid_attch_motif_id_pairs = torch.cat([valid_attch_motif_id_pairs, typed_attch_motif_id_pairs])

            #IDs of the available attachment atoms in the entire molecule's hgraph.
            attch_frontier_motif_mol_ids = motif_to_mol_atom_ids[frontier_motif_node_id][attch_frontier_motif_ids]
            attch_new_motif_mol_ids = motif_to_mol_atom_ids[new_motif_node_id][attch_new_motif_ids]
            typed_attch_mol_id_pairs = torch.cartesian_prod(attch_frontier_motif_mol_ids, attch_new_motif_mol_ids)
            valid_attch_mol_id_pairs = torch.cat([valid_attch_mol_id_pairs, typed_attch_mol_id_pairs])

            num_valid_attch_id_pairs_per_type.append(len(typed_attch_mol_id_pairs))
        
        valid_attch_rep_pairs = pred_hgraph.nodes["atom"].data["rep"][valid_attch_mol_id_pairs]
        #Get embeddings of the IDs (positions) of the atoms in their motifs,
        #so that atom's of the same type can be distinguished.
        valid_attch_motif_id_embed_pairs = self.embeddors["nodes"]["position"](valid_attch_motif_id_pairs)
        
        return (num_valid_attch_id_pairs_per_type, valid_attch_motif_id_pairs,
                valid_attch_mol_id_pairs, valid_attch_rep_pairs,
                valid_attch_motif_id_embed_pairs)

    def get_new_atom_attachments(self, ctx, step_ctx, targets, losses):
        """
        Get the index of a pair of atoms from the new motif
        and the frontier motif to attach them at.
        """

        inputs = ctx["input"][step_ctx["pred_to_batch_idx"]]
        #Dims: graph i X attch pair j X [attch atom 1, attch atom 2] X [atom type rep, atom motif id rep] X embed dim k.   
        valid_attch_rep_pairs_tensor = torch.zeros(
            len(inputs), step_ctx["max_attch_pairs"], 2, 2,
            self.node_rep_size, device = self.device
        )
        rep_iterator = enumerate(
            zip(step_ctx["valid_attch_rep_pairs"], step_ctx["valid_attch_motif_id_embed_pairs"])
        )
        for pred_idx, (valid_attch_rep_pairs, valid_attch_motif_id_embed_pairs) in rep_iterator:
            valid_attch_rep_pairs_tensor[pred_idx, :len(valid_attch_rep_pairs), 0] = (
                valid_attch_rep_pairs
            )
            valid_attch_rep_pairs_tensor[pred_idx, :len(valid_attch_rep_pairs), 1] = (
                valid_attch_motif_id_embed_pairs
            )

        attch_pairs_logits = self.attch_predictor(valid_attch_rep_pairs_tensor, inputs)
        if targets is not None:
            attch_pairs_log_probs = torch.nn.functional.log_softmax(attch_pairs_logits, dim = 1)
            target_attch_pair_idxs = [
                (step_ctx["valid_attch_motif_id_pairs"][pred_idx].tolist()
                 .index(targets["attch_pair"][batch_idx].tolist()))
                for pred_idx, batch_idx in enumerate(step_ctx["pred_to_batch_idx"])
            ]
            pred_losses = self.loss_func(attch_pairs_log_probs,
                                         torch.tensor(target_attch_pair_idxs, device = self.device))
            losses.extend(pred_losses)
            chosen_pair_idxs = target_attch_pair_idxs
        else:
            for pred_idx, valid_attch_rep_pairs in enumerate(step_ctx["valid_attch_rep_pairs"]):
                attch_pairs_logits[pred_idx][len(valid_attch_rep_pairs):] = float("-inf")
            attch_pairs_probs = torch.softmax(attch_pairs_logits, dim = 1)
            chosen_pair_idxs = list(torch.distributions.Categorical(attch_pairs_probs).sample())
        return chosen_pair_idxs
    
    def attach_new_atom_node(self, b, ctx, p, step_ctx):
        """
        Attachment pairs are pairs of attch atoms of the same type
        from the attch configs of the frontier motif and the new motif.
        When an attch pair is chosen, the two motifs will be attached by
        replacing the attch atom in the new motif with the attch
        atom from the frontier motif, i.e. transfering the new attch atom's edges 
        to the frontier motif's attch atom and deleting its node.
        """
        
        attch_frontier_motif_mol_id, attch_new_motif_mol_id = (
            step_ctx["valid_attch_mol_id_pairs"][p]
                    [step_ctx["chosen_pair_idx"][p]]
        )
        attch_frontier_motif_id, attch_new_motif_id = (
            step_ctx["valid_attch_motif_id_pairs"][p]
            [step_ctx["chosen_pair_idx"][p]]
        )
        self.merge_attachment_atoms(
            b, ctx, p, step_ctx,
            attch_frontier_motif_mol_id, attch_new_motif_mol_id
        ) 
        attch_type = self.get_chosen_attch_type(
            p, step_ctx
        )
      
        self.update_availability(
            b, ctx, p, step_ctx,
            attch_frontier_motif_id, attch_new_motif_id, attch_type
        )
       
        chosen_pair = (step_ctx["valid_attch_motif_id_pairs"][p]
                               [step_ctx["chosen_pair_idx"][p]])
        (ctx["pred_hgraph"][b].edges[("motif", "attaches to", "motif")]
         .data["attachment_motif_id_pair"][-1]) = chosen_pair
        (ctx["pred_hgraph"][b].edges[("attachment_config", "attaches to", "attachment_config")]
         .data["attachment_motif_id_pair"][-1]) = chosen_pair

    def merge_attachment_atoms(self, b, ctx, p, step_ctx,
                               attch_frontier_motif_mol_id, attch_new_motif_mol_id):
        """
        Replace the new motif's attch atom with the frontier motif's attch atom
        in the hgraph.
        
        Args:
        attch_frontier_motif_mol_id - 
            The ID of the attachment atom of the frontier motif in the hgraph.
        attch_new_motif_id - 
            The ID of the attachment atom of the new motif in the hgraph.
        """

        pred_hgraph = ctx["pred_hgraph"][b]

        edge_data = pred_hgraph.edges[("atom", "bond", "atom")].data
        
        #Replace bonds going out of the new motif's attachment atom.
        out_edge_src, out_edge_dst, out_edge_ids = pred_hgraph.out_edges(
            [attch_new_motif_mol_id], form = "all", etype = ("atom", "bond", "atom")
        )
        attch_out_edge_src = attch_frontier_motif_mol_id.repeat(len(out_edge_ids))
        out_edge_data = {key: edge_data[key][out_edge_ids] for key in edge_data.keys()}
        pred_hgraph.add_edges(attch_out_edge_src, out_edge_dst,
                                  out_edge_data, ("atom", "bond", "atom"))

        #Replace (atom, of, attachment_config) edge going out of the new motif's attachment atom.
        pred_hgraph.add_edges(attch_frontier_motif_mol_id,
                              step_ctx["new_attch_conf_node_id"][p],
                              etype = ("atom", "of", "attachment_config"))
                             
        #Replace bonds going into the new motif's attachment atom.
        in_edge_src, in_edge_dst, in_edge_ids = pred_hgraph.in_edges(
            [attch_new_motif_mol_id], form = "all", etype = ("atom", "bond", "atom")
        )
        in_edge_data = {key: edge_data[key][in_edge_ids] for key in edge_data.keys()}
        attch_in_edge_dst = attch_frontier_motif_mol_id.repeat(len(in_edge_ids))
        pred_hgraph.add_edges(in_edge_src, attch_in_edge_dst,
                                  in_edge_data, ("atom", "bond", "atom"))
        
        #Delete the new motif's attachment atom's node.
        pred_hgraph.remove_nodes([attch_new_motif_mol_id], ntype = "atom")
        
        #DGL reindexes nodes after removing one, so update the motif-to-mol atom id map.
        new_motif_node_id = step_ctx["new_motif_node_id"][p]
        ctx["motif_to_mol_atom_ids"][b][new_motif_node_id] = torch.tensor([
          atom_id if atom_id < attch_new_motif_mol_id else
          attch_frontier_motif_mol_id if atom_id == attch_new_motif_mol_id else
          atom_id - 1
          for atom_id in ctx["motif_to_mol_atom_ids"][b][new_motif_node_id]
        ], device = self.device)
    
    def get_chosen_attch_type(self, p, step_ctx):
        for i, type_ in enumerate(step_ctx["common_attch_types"][p]):
            last_idx_of_type = sum(step_ctx["n_valid_attch_pairs_per_type"][p][:i+1]) - 1
            if step_ctx["chosen_pair_idx"][p] <= last_idx_of_type:
                return type_

    def update_availability(self, b, ctx, p, step_ctx,
                            attch_frontier_motif_id, attch_new_motif_id,
                            attch_type):
        """
        Mark selected attachments as unavailable in their respective motifs.

        Args:
        attch_frontier_motif_id - The ID of the attachment atom in the frontier motif.
        attch_new_motif_id - The ID of the attachment atom in the new motif.
        """
        
        frontier_motif_node_id = ctx["frontier_node_ids"][b][-1].motif
        ctx["avail_attchs"][b][frontier_motif_node_id][attch_type].remove(attch_frontier_motif_id)
        if len(ctx["avail_attchs"][b][frontier_motif_node_id][attch_type]) == 0:
            del ctx["avail_attchs"][b][frontier_motif_node_id][attch_type]
        
        new_motif_node_id = step_ctx["new_motif_node_id"][p] 
        ctx["avail_attchs"][b][new_motif_node_id][attch_type].remove(attch_new_motif_id)
        if len(ctx["avail_attchs"][b][new_motif_node_id][attch_type]) == 0:
            del ctx["avail_attchs"][b][new_motif_node_id][attch_type]
