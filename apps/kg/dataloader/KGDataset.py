import os

def _download_and_extract(url, path, filename):
    import shutil, zipfile
    from tqdm import tqdm
    import requests

    fn = os.path.join(path, filename)

    while True:
        try:
            with zipfile.ZipFile(fn) as zf:
                zf.extractall(path)
            print('Unzip finished.')
            break
        except Exception:
            os.makedirs(path, exist_ok=True)
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(fn, 'wb') as writer:
                for chunk in tqdm(f_remote.iter_content(chunk_size=1024*1024)):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')

class KGDataset1:
    '''Load a knowledge graph with format 1

    In this format, the folder with a knowledge graph has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''
    def __init__(self, path, name):
        url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        path = os.path.join(path, name)

        with open(os.path.join(path, 'entities.dict')) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        self.entity2id = entity2id

        with open(os.path.join(path, 'relations.dict')) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        self.relation2id = relation2id

        # TODO: to deal with contries dataset.

        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)

        self.train = self.read_triple(path, 'train')
        self.valid = self.read_triple(path, 'valid')
        self.test = self.read_triple(path, 'test')

    def read_triple(self, path, mode):
        # mode: train/valid/test
        triples = []
        with open(os.path.join(path, '{}.txt'.format(mode))) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

        return triples


class KGDataset2:
    '''Load a knowledge graph with format 2

    In this format, the folder with a knowledge graph has five files:
    * entity2id.txt stores the mapping between entity name and entity Id.
    * relation2id.txt stores the mapping between relation name relation Id.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.

    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name):
        url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, '{}.zip'.format(name))
        self.path = os.path.join(path, name)

        f_ent2id = os.path.join(self.path, 'entity2id.txt')
        f_rel2id = os.path.join(self.path, 'relation2id.txt')

        with open(f_ent2id) as f_ent:
            self.n_entities = int(f_ent.readline()[:-1])
        with open(f_rel2id) as f_rel:
            self.n_relations = int(f_rel.readline()[:-1])

        self.train = self.read_triple(self.path, 'train')
        self.valid = self.read_triple(self.path, 'valid')
        self.test = self.read_triple(self.path, 'test')

    def read_triple(self, path, mode, skip_first_line=False):
        triples = []
        print('Reading {} triples....'.format(mode))
        with open(os.path.join(path, '{}.txt'.format(mode))) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                h, t, r = line.strip().split('\t')
                triples.append((int(h), int(r), int(t)))
        print('Finished. Read {} {} triples.'.format(len(triples), mode))
        return triples


def get_dataset(data_path, data_name, format_str):
    if data_name == 'Freebase':
        dataset = KGDataset2(data_path, data_name)
    elif format_str == '1':
        dataset = KGDataset1(data_path, data_name)
    else:
        dataset = KGDataset2(data_path, data_name)

    return dataset
