"""Generate vocabulary for a new dataset."""
if __name__ == '__main__':
    import argparse
    import os
    import rdkit

    from dgl.data.utils import _get_dgl_url, download, get_download_dir, extract_archive

    from jtnn.mol_tree import DGLMolTree

    parser = argparse.ArgumentParser('Generate vocabulary for a molecule dataset')
    parser.add_argument('-d', '--data-path', type=str,
                        help='Path to the dataset')
    parser.add_argument('-v', '--vocab', type=str,
                        help='Path to the vocabulary file to save')
    args = parser.parse_args()

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = set()
    with open(args.data_path, 'r') as f:
        for line in f:
            smiles = line.strip()
            mol = DGLMolTree(smiles)
            for i in mol.nodes_dict:
                vocab.add(mol.nodes_dict[i]['smiles'])

    with open(args.vocab, 'w') as f:
        for v in vocab:
            f.write(v + '\n')

    # Get the vocabulary used for the pre-trained model
    default_dir = get_download_dir()
    vocab_file = '{}/jtnn/{}.txt'.format(default_dir, 'vocab')
    if not os.path.exists(vocab_file):
        zip_file_path = '{}/jtnn.zip'.format(default_dir)
        download(_get_dgl_url('dgllife/jtnn.zip'), path=zip_file_path)
        extract_archive(zip_file_path, '{}/jtnn'.format(default_dir))
    default_vocab = set()
    with open(vocab_file, 'r') as f:
        for line in f:
            default_vocab.add(line.strip())

    print('The new vocabulary is a subset of the default vocabulary: {}'.format(
        vocab.issubset(default_vocab)))
