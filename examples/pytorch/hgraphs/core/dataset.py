from pathlib import Path
import dgl
import core.preprocess as preprocess

class MolDataset(dgl.data.DGLDataset):
    def __init__(self, name, data_root_path, raw_dir, save_dir,
                 smiles_file_name, motif_min_occurences = 0):
        self.data_root_path = data_root_path 
        self.smiles_file_name = smiles_file_name
        self.vocabs = {}
        self.mol_hgraphs = []
        self.motif_graphs = []
        self.motif_min_occurences = motif_min_occurences
        
        super().__init__(
            name = name, raw_dir = raw_dir, save_dir = save_dir,
            url = None, force_reload = False, verbose = True
        )
    
    def process(self):
        smiles_file_path = str(self.raw_path/self.smiles_file_name)
        self.mol_SMILES, self.vocabs, self.motif_graphs, self.mol_hgraphs = (
            preprocess.preprocess_mols_data(smiles_file_path, self.motif_min_occurences)
        )

    def save(self):
        dgl.data.utils.save_info(str(self.save_path/"mol_SMILES.pkl"), self.mol_SMILES)
        dgl.data.utils.save_info(str(self.save_path/"vocabs.pkl"), self.vocabs)
        dgl.data.utils.save_graphs(str(self.save_path/"motif_graphs.bin"), self.motif_graphs)
        dgl.data.utils.save_graphs(str(self.save_path/"mol_hgraphs.bin"), self.mol_hgraphs)

    def load(self):
        self.mol_SMILES = dgl.data.utils.load_info(str(self.save_path/"mol_SMILES.pkl"))
        self.vocabs = dgl.data.utils.load_info(str(self.save_path/"vocabs.pkl"))
        self.motif_graphs, self.__motif_labels = dgl.data.utils.load_graphs(str(self.save_path/"motif_graphs.bin"))
        self.mol_hgraphs, self.__mol_labels = dgl.data.utils.load_graphs(str(self.save_path/"mol_hgraphs.bin"))

    def has_cache(self):
        datafile_names =  ["mol_hgraphs.bin", "motif_graphs.bin", "vocabs.pkl", "mol_SMILES.pkl"]
        for datafile_name in datafile_names:
            datafile_path = self.save_path/datafile_name

            if not datafile_path.is_file():
                return False
        return True

    def __getitem__(self, idx):
        return self.mol_hgraphs[idx]

    def __len__(self):
        return len(self.mol_hgraphs)

    @property
    def raw_path(self):
        return Path(self.data_root_path)/self.name/self.raw_dir
    
    @property
    def save_path(self):
        return Path(self.data_root_path)/self.name/self.save_dir

def collate_(batch, device):
    """
    The collation function for dataloaders to use.
    Should be made partial wrt the device context,
    i.e. collate = lambda batch: collate_(batch, "cuda").
    """

    for i, graph in enumerate(batch):
        batch[i] = graph.to(device)

    return batch
