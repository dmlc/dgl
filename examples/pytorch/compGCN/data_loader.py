import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import argparse
import os

class TrainDataset(Dataset):
	"""
	Training Dataset class.
	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= label
			mask		= np.ones([self.p.num_ent], dtype=np.bool)
			mask[label]	= 0
			neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
			neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class TestDataset(Dataset):
	"""
	Evaluation Dataset class.
	Parameters
	----------
	triples:	The triples used for evaluating the model
	params:		Parameters for the experiments
	
	Returns
	-------
	An evaluation Dataset class instance used by DataLoader for model evaluation
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class Data(object):

	def __init__(self, params):
		"""
		Reading in raw triples and converts it into a standard format. 
		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits
		"""
		self.p= params
		#read in raw data and get mappings
		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent = len(self.ent2id)
		self.p.num_rel = len(self.rel2id) // 2

		self.data = ddict(list)
		sr2o = ddict(set)
		#construct dgl graph
		src=[]
		dst=[]
		rels = []
		inver_src = []
		inver_dst = []
		inver_rels = []

		for split in ['train', 'test', 'valid']:
			for line in open('./{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
					src.append(sub)
					dst.append(obj)
					rels.append(rel)
					inver_src.append(obj)
					inver_dst.append(sub)
					inver_rels.append(rel+self.p.num_rel)

		#identift in and out edges
		src = src + inver_src
		dst = dst + inver_dst
		rels = rels + inver_rels
		self.g = dgl.graph((src, dst), num_nodes=self.p.num_ent)
		self.g.edata['etype'] = torch.Tensor(rels).long()
		in_edges_mask = [True] * (self.g.num_edges()//2) + [False] * (self.g.num_edges()//2)
		out_edges_mask = [False] * (self.g.num_edges()//2) + [True] * (self.g.num_edges()//2)
		self.g.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
		self.g.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)
		
		#Prepare train/valid/test data
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)
		#train/valid/test dataloaders
		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}
