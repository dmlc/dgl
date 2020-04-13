.. _apimodelpretrain:

Pre-trained Models
==================

We provide multiple pre-trained models for users to use without the need of training from scratch.

Example Usage
-------------

Property Prediction
```````````````````

.. code-block:: python

    from dgllife.data import Tox21
    from dgllife.model import load_pretrained
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    dataset = Tox21(smiles_to_bigraph, CanonicalAtomFeaturizer())
    model = load_pretrained('GCN_Tox21') # Pretrained model loaded
    model.eval()

    smiles, g, label, mask = dataset[0]
    feats = g.ndata.pop('h')
    label_pred = model(g, feats)
    print(smiles)                   # CCOc1ccc2nc(S(N)(=O)=O)sc2c1
    print(label_pred[:, mask != 0]) # Mask non-existing labels
    # tensor([[ 1.4190, -0.1820,  1.2974,  1.4416,  0.6914,
    # 2.0957,  0.5919,  0.7715, 1.7273,  0.2070]])

Generative Models

.. code-block:: python

    from dgllife.model import load_pretrained

    model = load_pretrained('DGMG_ZINC_canonical')
    model.eval()
    smiles = []
    for i in range(4):
        smiles.append(model(rdkit_mol=True))

    print(smiles)
    # ['CC1CCC2C(CCC3C2C(NC2=CC(Cl)=CC=C2N)S3(=O)=O)O1',
    # 'O=C1SC2N=CN=C(NC(SC3=CC=CC=N3)C1=CC=CO)C=2C1=CCCC1',
    # 'CC1C=CC(=CC=1)C(=O)NN=C(C)C1=CC=CC2=CC=CC=C21',
    # 'CCN(CC1=CC=CC=C1F)CC1CCCN(C)C1']

If you are running the code block above in Jupyter notebook, you can also visualize the molecules generated with

.. code-block:: python

    from IPython.display import SVG
    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    SVG(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(180, 150), useSVG=True))

.. image:: https://data.dgl.ai/dgllife/dgmg/dgmg_model_zoo_example2.png

API
---

.. autofunction:: dgllife.model.load_pretrained
