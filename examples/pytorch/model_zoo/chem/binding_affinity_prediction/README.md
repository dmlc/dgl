# Binding Affinity Prediction

## Datasets
- **PDBBind**: The PDBBind dataset in MoleculeNet [1] processed from the PDBBind database. The PDBBind 
database consists of experimentally measured binding affinities for bio-molecular complexes [2], [3]. 
It provides detailed 3D Cartesian coordinates of both ligands and their target proteins derived from 
experimental(e.g., X-ray crystallography) measurements. The availability of coordinates of the 
protein-ligand complexes permits structure-based featurization that is aware of the protein-ligand 
binding geometry. The authors of [1] use the "refined" and "core" subsets of the database [4], more carefully 
processed for data artifacts, as additional benchmarking targets.

## Models
- **Atomic Convolutional Networks (ACNN)** [5]: Constructs nearest neighbor graphs separately for the ligand, protein and complex 
based on the 3D coordinates of the atoms and predicts the binding free energy.

## Usage

Use `main.py` with arguments
```
-m {ACNN}, Model to use
-d {PDBBind_core_pocket_random, PDBBind_core_pocket_scaffold, PDBBind_core_pocket_stratified, 
PDBBind_core_pocket_temporal, PDBBind_refined_pocket_random, PDBBind_refined_pocket_scaffold, 
PDBBind_refined_pocket_stratified, PDBBind_refined_pocket_temporal}, dataset and splitting method to use
```

## Performance

### PDBBind

#### ACNN

| Subset  | Splitting Method | Test MAE | Test R2 |
| ------- | ---------------- | -------- | ------- |
| Core    | Random           | 1.7688   | 0.1511  |
| Core    | Scaffold         | 2.5420   | 0.1471  |
| Core    | Stratified       | 1.7419   | 0.1520  |
| Core    | Temporal         | 1.9543   | 0.1640  |
| Refined | Random           | 1.1948   | 0.4373  |    
| Refined | Scaffold         | 1.4021   | 0.2086  |
| Refined | Stratified       | 1.6376   | 0.3050  |
| Refined | Temporal         | 1.2457   | 0.3438  |

## Speed

### ACNN

Comparing to the [DeepChem's implementation](https://github.com/joegomes/deepchem/tree/acdc), we achieve a speedup by 
roughly 3.3 for training time per epoch (from 1.40s to 0.42s). If we do not care about 
randomness introduced by some kernel optimization, we can achieve a speedup by roughly 4.4 (from 1.40s to 0.32s).

## References
[1] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.
[2] Wang et al. (2004) The PDBbind database: collection of binding affinities for protein-ligand complexes 
with known three-dimensional structures. *J Med Chem* 3;47(12):2977-80.
[3] Wang et al. (2005) The PDBbind database: methodologies and updates. *J Med Chem* 16;48(12):4111-9.
[4] Liu et al. (2015) PDB-wide collection of binding data: current status of the PDBbind database. *Bioinformatics* 1;31(3):405-12.
[5] Gomes et al. (2017) Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity. *arXiv preprint arXiv:1703.10603*.
