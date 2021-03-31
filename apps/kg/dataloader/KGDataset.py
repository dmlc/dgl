# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import numpy as np

def _download_and_extract(url, path, filename):
    import shutil, zipfile
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
                for chunk in f_remote.iter_content(chunk_size=1024*1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def _parse_srd_format(format):
    if format == "hrt":
        return [0, 1, 2]
    if format == "htr":
        return [0, 2, 1]
    if format == "rht":
        return [1, 0, 2]
    if format == "rth":
        return [2, 0, 1]
    if format == "thr":
        return [1, 2, 0]
    if format == "trh":
        return [2, 1, 0]

def _file_line(path):
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class KGDataset:
    '''Load a knowledge graph

    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * test stores the triples in the test set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''
    def __init__(self, entity_path, relation_path, train_path, 
                 valid_path=None, test_path=None, format=[0,1,2], skip_first_line=False):

        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line, format)
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", skip_first_line, format)

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=[0,1,2]):
        # mode: train/valid/test
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split('\t')
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return (heads, rels, tails)


class PartitionKGDataset():
    '''Load a partitioned knowledge graph

    The folder with a partitioned knowledge graph has four files:
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * local_to_global stores the mapping of local id and global id
    * partition_book stores the machine id of each entity

    The triples are stored as 'head_id\relation_id\tail_id'.
    '''
    def __init__(self, relation_path, train_path, local2global_path, 
                 read_triple=True, skip_first_line=False):
        self.n_entities = _file_line(local2global_path)
        if skip_first_line == False:
            self.n_relations = _file_line(relation_path)
        else:
            self.n_relations = _file_line(relation_path) - 1
        if read_triple == True:
            self.train = self.read_triple(train_path, "train")

    def read_triple(self, path, mode):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                heads.append(int(h))
                rels.append(int(r))
                tails.append(int(t))

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return (heads, rels, tails)


class KGDatasetFB15k(KGDataset):
    '''Load a knowledge graph FB15k

    The FB15k dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='FB15k'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetFB15k, self).__init__(os.path.join(self.path, 'entities.dict'),
                                             os.path.join(self.path, 'relations.dict'),
                                             os.path.join(self.path, 'train.txt'),
                                             os.path.join(self.path, 'valid.txt'),
                                             os.path.join(self.path, 'test.txt'))


class KGDatasetFB15k237(KGDataset):
    '''Load a knowledge graph FB15k-237

    The FB15k-237 dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='FB15k-237'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetFB15k237, self).__init__(os.path.join(self.path, 'entities.dict'),
                                                os.path.join(self.path, 'relations.dict'),
                                                os.path.join(self.path, 'train.txt'),
                                                os.path.join(self.path, 'valid.txt'),
                                                os.path.join(self.path, 'test.txt'))


class KGDatasetWN18(KGDataset):
    '''Load a knowledge graph wn18

    The wn18 dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='wn18'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetWN18, self).__init__(os.path.join(self.path, 'entities.dict'),
                                            os.path.join(self.path, 'relations.dict'),
                                            os.path.join(self.path, 'train.txt'),
                                            os.path.join(self.path, 'valid.txt'),
                                            os.path.join(self.path, 'test.txt'))


class KGDatasetWN18rr(KGDataset):
    '''Load a knowledge graph wn18rr

    The wn18rr dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='wn18rr'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetWN18rr, self).__init__(os.path.join(self.path, 'entities.dict'),
                                              os.path.join(self.path, 'relations.dict'),
                                              os.path.join(self.path, 'train.txt'),
                                              os.path.join(self.path, 'valid.txt'),
                                              os.path.join(self.path, 'test.txt'))

class KGDatasetFreebase(KGDataset):
    '''Load a knowledge graph Full Freebase

    The Freebase dataset has five files:
    * entity2id.txt stores the mapping between entity name and entity Id.
    * relation2id.txt stores the mapping between relation name relation Id.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='Freebase'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, '{}.zip'.format(name))
        self.path = os.path.join(path, name)

        super(KGDatasetFreebase, self).__init__(os.path.join(self.path, 'entity2id.txt'),
                                                os.path.join(self.path, 'relation2id.txt'),
                                                os.path.join(self.path, 'train.txt'),
                                                os.path.join(self.path, 'valid.txt'),
                                                os.path.join(self.path, 'test.txt'))

    def read_entity(self, entity_path):
        with open(entity_path) as f_ent:
            n_entities = int(f_ent.readline()[:-1])
        return None, n_entities

    def read_relation(self, relation_path):
        with open(relation_path) as f_rel:
            n_relations = int(f_rel.readline()[:-1])
        return None, n_relations

    def read_triple(self, path, mode, skip_first_line=False, format=None):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                h, t, r = line.strip().split('\t')
                heads.append(int(h))
                tails.append(int(t))
                rels.append(int(r))

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        return (heads, rels, tails)

class KGDatasetUDDRaw(KGDataset):
    '''Load a knowledge graph user defined dataset

    The user defined dataset has five files:
    * entities stores the mapping between entity name and entity Id.
    * relations stores the mapping between relation name relation Id.
    * train stores the triples in the training set. In format [src_name, rel_name, dst_name]
    * valid stores the triples in the validation set. In format [src_name, rel_name, dst_name]
    * test stores the triples in the test set. In format [src_name, rel_name, dst_name]

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name, files, format):
        self.name = name
        for f in files:
            assert os.path.exists(os.path.join(path, f)), \
                'File {} now exist in {}'.format(f, path)

        assert len(format) == 3
        format = _parse_srd_format(format)
        self.load_entity_relation(path, files, format)

        # Only train set is provided
        if len(files) == 1:
            super(KGDatasetUDDRaw, self).__init__("entities.tsv",
                                                  "relation.tsv",
                                                  os.path.join(path, files[0]),
                                                  format=format)
        # Train, validation and test set are provided
        if len(files) == 3:
            super(KGDatasetUDDRaw, self).__init__("entities.tsv",
                                                  "relation.tsv",
                                                  os.path.join(path, files[0]),
                                                  os.path.join(path, files[1]),
                                                  os.path.join(path, files[2]),
                                                  format=format)

    def load_entity_relation(self, path, files, format):
        entity_map = {}
        rel_map = {}
        for fi in files:
            with open(os.path.join(path, fi)) as f:
                for line in f:
                    triple = line.strip().split('\t')
                    src, rel, dst = triple[format[0]], triple[format[1]], triple[format[2]]
                    src_id = _get_id(entity_map, src)
                    dst_id = _get_id(entity_map, dst)
                    rel_id = _get_id(rel_map, rel)

        entities = ["{}\t{}\n".format(key, val) for key, val in entity_map.items()]
        with open(os.path.join(path, "entities.tsv"), "w+") as f:
            f.writelines(entities)
        self.entity2id = entity_map
        self.n_entities = len(entities)

        relations = ["{}\t{}\n".format(key, val) for key, val in rel_map.items()]
        with open(os.path.join(path, "relations.tsv"), "w+") as f:
            f.writelines(relations)
        self.relation2id = rel_map
        self.n_relations = len(relations)

    def read_entity(self, entity_path):
        return self.entity2id, self.n_entities
    
    def read_relation(self, relation_path):
        return self.relation2id, self.n_relations

class KGDatasetUDD(KGDataset):
    '''Load a knowledge graph user defined dataset

    The user defined dataset has five files:
    * entities stores the mapping between entity name and entity Id.
    * relations stores the mapping between relation name relation Id.
    * train stores the triples in the training set. In format [src_id, rel_id, dst_id]
    * valid stores the triples in the validation set. In format [src_id, rel_id, dst_id]
    * test stores the triples in the test set. In format [src_id, rel_id, dst_id]

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name, files, format):
        self.name = name
        for f in files:
            assert os.path.exists(os.path.join(path, f)), \
                'File {} now exist in {}'.format(f, path)

        format = _parse_srd_format(format)
        if len(files) == 3:
            super(KGDatasetUDD, self).__init__(os.path.join(path, files[0]),
                                               os.path.join(path, files[1]),
                                               os.path.join(path, files[2]),
                                               None, None,
                                               format=format)
        if len(files) == 5:
            super(KGDatasetUDD, self).__init__(os.path.join(path, files[0]),
                                               os.path.join(path, files[1]),
                                               os.path.join(path, files[2]),
                                               os.path.join(path, files[3]),
                                               os.path.join(path, files[4]),
                                               format=format)

    def read_entity(self, entity_path):
        n_entities = 0
        with open(entity_path) as f_ent:
            for line in f_ent:
                n_entities += 1
        return None, n_entities

    def read_relation(self, relation_path):
        n_relations = 0
        with open(relation_path) as f_rel:
            for line in f_rel:
                n_relations += 1
        return None, n_relations

    def read_triple(self, path, mode, skip_first_line=False, format=[0,1,2]):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split('\t')
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(int(h))
                tails.append(int(t))
                rels.append(int(r))
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        return (heads, rels, tails)

def get_dataset(data_path, data_name, format_str, files=None):
    if format_str == 'built_in':
        if data_name == 'Freebase':
            dataset = KGDatasetFreebase(data_path)
        elif data_name == 'FB15k':
            dataset = KGDatasetFB15k(data_path)
        elif data_name == 'FB15k-237':
            dataset = KGDatasetFB15k237(data_path)
        elif data_name == 'wn18':
            dataset = KGDatasetWN18(data_path)
        elif data_name == 'wn18rr':
            dataset = KGDatasetWN18rr(data_path)
        else: 
            assert False, "Unknown dataset {}".format(data_name)
    elif format_str.startswith('raw_udd'):
        # user defined dataset
        format = format_str[8:]
        dataset = KGDatasetUDDRaw(data_path, data_name, files, format)
    elif format_str.startswith('udd'):
        # user defined dataset
        format = format_str[4:]
        dataset = KGDatasetUDD(data_path, data_name, files, format)
    else:
        assert False, "Unknown format {}".format(format_str)

    return dataset


def get_partition_dataset(data_path, data_name, part_id):
    part_name = os.path.join(data_name, 'partition_'+str(part_id))
    path = os.path.join(data_path, part_name)

    if not os.path.exists(path):
        print('Partition file not found.')
        exit()

    train_path = os.path.join(path, 'train.txt')
    local2global_path = os.path.join(path, 'local_to_global.txt')
    partition_book_path = os.path.join(path, 'partition_book.txt')

    if data_name == 'Freebase':
        relation_path = os.path.join(path, 'relation2id.txt')
        skip_first_line = True
    elif data_name in ['FB15k', 'FB15k-237', 'wn18', 'wn18rr']:
        relation_path = os.path.join(path, 'relations.dict')
        skip_first_line = False
    else:
        relation_path = os.path.join(path, 'relation.tsv')
        skip_first_line = False

    dataset = PartitionKGDataset(relation_path, 
                                 train_path, 
                                 local2global_path, 
                                 read_triple=True, 
                                 skip_first_line=skip_first_line)

    partition_book = []
    with open(partition_book_path) as f:
        for line in f:
            partition_book.append(int(line))

    local_to_global = []
    with open(local2global_path) as f:
        for line in f:
            local_to_global.append(int(line))

    return dataset, partition_book, local_to_global


def get_server_partition_dataset(data_path, data_name, part_id):
    part_name = os.path.join(data_name, 'partition_'+str(part_id))
    path = os.path.join(data_path, part_name)

    if not os.path.exists(path):
        print('Partition file not found.')
        exit()

    train_path = os.path.join(path, 'train.txt')
    local2global_path = os.path.join(path, 'local_to_global.txt')    

    if data_name == 'Freebase':
        relation_path = os.path.join(path, 'relation2id.txt')
        skip_first_line = True
    elif data_name in ['FB15k', 'FB15k-237', 'wn18', 'wn18rr']:
        relation_path = os.path.join(path, 'relations.dict')
        skip_first_line = False
    else:
        relation_path = os.path.join(path, 'relation.tsv')
        skip_first_line = False

    dataset = PartitionKGDataset(relation_path,
                                 train_path,
                                 local2global_path,
                                 read_triple=False,
                                 skip_first_line=skip_first_line)

    n_entities = _file_line(os.path.join(path, 'partition_book.txt'))

    local_to_global = []
    with open(local2global_path) as f:
        for line in f:
            local_to_global.append(int(line))

    global_to_local = [0] * n_entities
    for i in range(len(local_to_global)):
        global_id = local_to_global[i]
        global_to_local[global_id] = i

    local_to_global = None

    return global_to_local, dataset
