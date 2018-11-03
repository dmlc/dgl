from jtnn.mpn import mol2dgl, DGLMPN
from official.mpn import MPN, mol2graph
from jtnn.jtnn_enc import DGLJTNNEncoder
from official.jtnn_enc import JTNNEncoder
from jtnn.jtnn_dec import DGLJTNNDecoder
from official.jtnn_dec import JTNNDecoder
from jtnn.jtmpn import DGLJTMPN
from official.jtmpn import JTMPN
from official.mol_tree import MolTree, Vocab
from jtnn.mol_tree_nx import DGLMolTree
from jtnn.jtnn_vae import dgl_set_batch_nodeID, DGLJTNNVAE
from official.jtnn_vae import set_batch_nodeID, JTNNVAE
import torch
from torch import nn
from dgl import batch
import rdkit
from rdkit import Chem
import numpy as np

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

smiles_batch = '''
CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1
COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1
C=CCn1c(S[C@H](C)c2nc3sc(C)c(C)c3c(=O)[nH]2)nnc1C1CC1
C[NH+](C/C=C/c1ccco1)CCC(F)(F)F
COc1ccc(N2C(=O)C(=O)N(CN3CCC(c4nc5ccccc5s4)CC3)C2=O)cc1
Cc1ccc([C@@H](C)[NH2+][C@H](C)C(=O)Nc2ccccc2F)cc1
O=c1cc(C[NH2+]Cc2cccc(Cl)c2)nc(N2CCCC2)[nH]1
O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O
'''.strip().split()
#smiles_batch = '''
#c1ccccc1
#'''.strip().split()

def allclose(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-7)

def isclose(a, b):
    return torch.isclose(a, b, rtol=1e-4, atol=1e-7)


def test_mpn():
    gl = mol2dgl(smiles_batch)
    dglmpn = DGLMPN(5, 4)
    mpn = MPN(5, 4)
    mpn.W_i = dglmpn.W_i
    mpn.W_o = dglmpn.gather_updater.W_o
    mpn.W_h = dglmpn.loopy_bp_updater.W_h

    glb = batch(gl)

    result = dglmpn.forward(gl)
    mol_vec = mpn(mol2graph(smiles_batch))

    assert allclose(result, mol_vec)


def test_treeenc():
    mol_batch = [MolTree(smiles) for smiles in smiles_batch]
    for mol_tree in mol_batch:
        mol_tree.recover()
        mol_tree.assemble()

    vocab = [x.strip('\r\n ') for x in open('data/vocab.txt')]
    vocab = Vocab(vocab)

    set_batch_nodeID(mol_batch, vocab)

    emb = nn.Embedding(vocab.size(), 5)
    jtnn = JTNNEncoder(vocab, 5, emb)

    root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
    tree_mess, tree_vec = jtnn(root_batch)

    nx_mol_batch = [DGLMolTree(smiles) for smiles in smiles_batch]
    for nx_mol_tree in nx_mol_batch:
        nx_mol_tree.recover()
        nx_mol_tree.assemble()
    dgl_set_batch_nodeID(nx_mol_batch, vocab)

    dgljtnn = DGLJTNNEncoder(vocab, 5, emb)
    dgljtnn.enc_tree_update.W_z = jtnn.W_z
    dgljtnn.enc_tree_update.W_h = jtnn.W_h
    dgljtnn.enc_tree_update.W_r = jtnn.W_r
    dgljtnn.enc_tree_update.U_r = jtnn.U_r
    dgljtnn.enc_tree_gather_update.W = jtnn.W

    mol_tree_batch, dgl_tree_vec = dgljtnn(nx_mol_batch)
    dgl_tree_mess = mol_tree_batch.edata['m']

    assert dgl_tree_mess.shape[0] == len(tree_mess)
    fail = False
    for u, v in tree_mess:
        eid = mol_tree_batch.edge_id(u, v)
        if not allclose(tree_mess[(u, v)], dgl_tree_mess[eid]):
            fail = True
            print(u, v, tree_mess[(u, v)], dgl_tree_mess[eid][0])
    assert not fail

    assert allclose(dgl_tree_vec, tree_vec)

    # Graph decoder
    cands = []
    dglcands = []
    jtmpn = JTMPN(5, 4)
    dgljtmpn = DGLJTMPN(5, 4)
    dgljtmpn.W_i = jtmpn.W_i
    dgljtmpn.gather_updater.W_o = jtmpn.W_o
    dgljtmpn.loopy_bp_updater.W_h = jtmpn.W_h

    for i, mol_tree in enumerate(mol_batch):
        for node in mol_tree.nodes:
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cand_mols])

    cand_vec = jtmpn(cands, tree_mess)

    for i, mol_tree in enumerate(nx_mol_batch):
        for node_id, node in mol_tree.nodes_dict.items():
            if node['is_leaf'] or len(node['cands']) == 1:
                continue
            dglcands.extend([
                (cand, mol_tree, node_id) for cand in node['cand_mols']
            ])

    assert len(cands) == len(dglcands)
    for item, dglitem in zip(cands, dglcands):
        assert Chem.MolToSmiles(item[0]) == Chem.MolToSmiles(dglitem[0])

    dgl_cand_vec = dgljtmpn(dglcands, mol_tree_batch)

    # TODO: add check.  Seems that the original implementation has a bug
    assert allclose(cand_vec, dgl_cand_vec)


def test_treedec():
    mol_batch = [MolTree(smiles) for smiles in smiles_batch]
    for mol_tree in mol_batch:
        mol_tree.recover()
        mol_tree.assemble()

    tree_vec = torch.randn(len(mol_batch), 5)
    vocab = [x.strip('\r\n ') for x in open('data/vocab.txt')]
    vocab = Vocab(vocab)

    set_batch_nodeID(mol_batch, vocab)

    nx_mol_batch = [DGLMolTree(smiles) for smiles in smiles_batch]
    for nx_mol_tree in nx_mol_batch:
        nx_mol_tree.recover()
        nx_mol_tree.assemble()
    dgl_set_batch_nodeID(nx_mol_batch, vocab)

    emb = nn.Embedding(vocab.size(), 5)
    dgljtnn = DGLJTNNDecoder(vocab, 5, 5, emb)
    dgl_q_loss, dgl_p_loss, dgl_q_acc, dgl_p_acc = dgljtnn(nx_mol_batch, tree_vec)

    jtnn = JTNNDecoder(vocab, 5, 5, emb)
    jtnn.W = dgljtnn.W
    jtnn.U = dgljtnn.U
    jtnn.W_o = dgljtnn.W_o
    jtnn.U_s = dgljtnn.U_s
    jtnn.W_z = dgljtnn.dec_tree_edge_update.W_z
    jtnn.W_r = dgljtnn.dec_tree_edge_update.W_r
    jtnn.U_r = dgljtnn.dec_tree_edge_update.U_r
    jtnn.W_h = dgljtnn.dec_tree_edge_update.W_h
    q_loss, p_loss, q_acc, p_acc = jtnn(mol_batch, tree_vec)

    assert isclose(p_loss, dgl_p_loss)
    assert isclose(q_loss, dgl_q_loss)
    assert isclose(p_acc, dgl_p_acc)
    assert isclose(q_acc, dgl_q_acc)


def test_vae():
    vocab = [x.strip('\r\n ') for x in open('data/vocab.txt')]
    vocab = Vocab(vocab)
    mol_batch = [MolTree(smiles) for smiles in smiles_batch]
    for mol_tree in mol_batch:
        mol_tree.recover()
        mol_tree.assemble()
    set_batch_nodeID(mol_batch, vocab)
    nx_mol_batch = [DGLMolTree(smiles) for smiles in smiles_batch]
    for nx_mol_tree in nx_mol_batch:
        nx_mol_tree.recover()
        nx_mol_tree.assemble()
    dgl_set_batch_nodeID(nx_mol_batch, vocab)

    vae = JTNNVAE(vocab, 50, 50, 3)
    dglvae = DGLJTNNVAE(vocab, 50, 50, 3)
    e1 = torch.randn(len(smiles_batch), 25)
    e2 = torch.randn(len(smiles_batch), 25)

    dglvae.embedding = vae.embedding
    dgljtnn, dgljtmpn, dglmpn, dgldecoder = dglvae.jtnn, dglvae.jtmpn, dglvae.mpn, dglvae.decoder
    jtnn, mpn, decoder, jtmpn = vae.jtnn, vae.mpn, vae.decoder, vae.jtmpn
    dgljtnn.enc_tree_update.W_z = jtnn.W_z
    dgljtnn.enc_tree_update.W_h = jtnn.W_h
    dgljtnn.enc_tree_update.W_r = jtnn.W_r
    dgljtnn.enc_tree_update.U_r = jtnn.U_r
    dgljtnn.enc_tree_gather_update.W = jtnn.W
    dgljtnn.embedding = jtnn.embedding
    dgljtmpn.W_i = jtmpn.W_i
    dgljtmpn.gather_updater.W_o = jtmpn.W_o
    dgljtmpn.loopy_bp_updater.W_h = jtmpn.W_h
    mpn.W_i = dglmpn.W_i
    mpn.W_o = dglmpn.gather_updater.W_o
    mpn.W_h = dglmpn.loopy_bp_updater.W_h
    decoder.W = dgldecoder.W
    decoder.U = dgldecoder.U
    decoder.W_o = dgldecoder.W_o
    decoder.U_s = dgldecoder.U_s
    decoder.W_z = dgldecoder.dec_tree_edge_update.W_z
    decoder.W_r = dgldecoder.dec_tree_edge_update.W_r
    decoder.U_r = dgldecoder.dec_tree_edge_update.U_r
    decoder.W_h = dgldecoder.dec_tree_edge_update.W_h
    decoder.embedding = dgldecoder.embedding
    dglvae.T_mean = vae.T_mean
    dglvae.G_mean = vae.G_mean
    dglvae.T_var = vae.T_var
    dglvae.G_var = vae.G_var

    loss, kl_loss, wacc, tacc, aacc, sacc = vae(mol_batch, e1=e1, e2=e2)
    loss_dgl, kl_loss_dgl, wacc_dgl, tacc_dgl, aacc_dgl, sacc_dgl = dglvae(nx_mol_batch, e1=e1, e2=e2)

    assert torch.allclose(loss, loss_dgl)
    assert torch.allclose(kl_loss, kl_loss_dgl)
    assert torch.allclose(wacc, wacc_dgl)
    assert torch.allclose(tacc, tacc_dgl)
    assert np.allclose(aacc, aacc_dgl)
    assert np.allclose(sacc, sacc_dgl)


if __name__ == '__main__':
    test_mpn()
    test_treeenc()
    test_treedec()
    test_vae()
