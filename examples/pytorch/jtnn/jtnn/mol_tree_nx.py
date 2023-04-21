import dgl
import numpy as np
import rdkit.Chem as Chem

from .chemutils import (
    decode_stereo,
    enum_assemble_nx,
    get_clique_mol,
    get_mol,
    get_smiles,
    set_atommap,
    tree_decomp,
)


class DGLMolTree(object):
    def __init__(self, smiles):
        self.nodes_dict = {}

        if smiles is None:
            self.graph = dgl.graph(([], []))
            return

        self.smiles = smiles
        self.mol = get_mol(smiles)

        # Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        # cliques: a list of list of atom indices
        cliques, edges = tree_decomp(self.mol)
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            csmiles = get_smiles(cmol)
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=get_mol(csmiles),
                clique=c,
            )
            if min(c) == 0:
                root = i

        # The clique with atom ID 0 becomes root
        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = (
                    self.nodes_dict[root][attr],
                    self.nodes_dict[0][attr],
                )

        src = np.zeros((len(edges) * 2,), dtype="int")
        dst = np.zeros((len(edges) * 2,), dtype="int")
        for i, (_x, _y) in enumerate(edges):
            x = 0 if _x == root else root if _x == 0 else _x
            y = 0 if _y == root else root if _y == 0 else _y
            src[2 * i] = x
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x
        self.graph = dgl.graph((src, dst), num_nodes=len(cliques))

        for i in self.nodes_dict:
            self.nodes_dict[i]["nid"] = i + 1
            if self.graph.out_degrees(i) > 1:  # Leaf node mol is not marked
                set_atommap(
                    self.nodes_dict[i]["mol"], self.nodes_dict[i]["nid"]
                )
            self.nodes_dict[i]["is_leaf"] = self.graph.out_degrees(i) == 1

    def treesize(self):
        return self.graph.num_nodes()

    def _recover_node(self, i, original_mol):
        node = self.nodes_dict[i]

        clique = []
        clique.extend(node["clique"])
        if not node["is_leaf"]:
            for cidx in node["clique"]:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(node["nid"])

        for j in self.graph.successors(i).numpy():
            nei_node = self.nodes_dict[j]
            clique.extend(nei_node["clique"])
            if nei_node["is_leaf"]:  # Leaf node, no need to mark
                continue
            for cidx in nei_node["clique"]:
                # allow singleton node override the atom mapping
                if cidx not in node["clique"] or len(nei_node["clique"]) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node["nid"])

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        node["label"] = Chem.MolToSmiles(
            Chem.MolFromSmiles(get_smiles(label_mol))
        )
        node["label_mol"] = get_mol(node["label"])

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return node["label"]

    def _assemble_node(self, i):
        neighbors = [
            self.nodes_dict[j]
            for j in self.graph.successors(i).numpy()
            if self.nodes_dict[j]["mol"].GetNumAtoms() > 1
        ]
        neighbors = sorted(
            neighbors, key=lambda x: x["mol"].GetNumAtoms(), reverse=True
        )
        singletons = [
            self.nodes_dict[j]
            for j in self.graph.successors(i).numpy()
            if self.nodes_dict[j]["mol"].GetNumAtoms() == 1
        ]
        neighbors = singletons + neighbors

        cands = enum_assemble_nx(self.nodes_dict[i], neighbors)

        if len(cands) > 0:
            (
                self.nodes_dict[i]["cands"],
                self.nodes_dict[i]["cand_mols"],
                _,
            ) = list(zip(*cands))
            self.nodes_dict[i]["cands"] = list(self.nodes_dict[i]["cands"])
            self.nodes_dict[i]["cand_mols"] = list(
                self.nodes_dict[i]["cand_mols"]
            )
        else:
            self.nodes_dict[i]["cands"] = []
            self.nodes_dict[i]["cand_mols"] = []

    def recover(self):
        for i in self.nodes_dict:
            self._recover_node(i, self.mol)

    def assemble(self):
        for i in self.nodes_dict:
            self._assemble_node(i)
