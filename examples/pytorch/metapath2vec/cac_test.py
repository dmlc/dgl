import logging
import sys
import os
import random
from collections import Counter
from time import strftime, localtime, time

import dgl
import os.path
from tqdm import tqdm
import torch
from torch import optim

from main2 import Metapath2vec


dirpath = "data/net_aminer"

class MetaPathGenerator:
	def __init__(self):
		self.id_author = dict()
		self.id_conf = dict()
		self.author_coauthorlist = dict()
		self.conf_authorlist = dict()
		self.author_conflist = dict()
		self.paper_author = dict()
		self.author_paper = dict()
		self.conf_paper = dict()
		self.paper_conf = dict()

	def read_data(self, dirpath):
		with open(dirpath + "/id_author.txt") as adictfile:
			for line in adictfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					self.id_author[int(toks[0])] = toks[1].replace(" ", "")

		#print "#authors", len(self.id_author)

		with open(dirpath + "/id_conf.txt") as cdictfile:
			for line in cdictfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					newconf = toks[1].replace(" ", "")
					self.id_conf[int(toks[0])] = newconf

		#print "#conf", len(self.id_conf)

		with open(dirpath + "/paper_author.txt") as pafile:
			for line in pafile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					p, a = toks[0], toks[1]
					if p not in self.paper_author:
						self.paper_author[p] = []
					self.paper_author[p].append(a)
					if a not in self.author_paper:
						self.author_paper[a] = []
					self.author_paper[a].append(p)

		with open(dirpath + "/paper_conf.txt") as pcfile:
			for line in pcfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					p, c = toks[0], toks[1]
					self.paper_conf[p] = c
					if c not in self.conf_paper:
						self.conf_paper[c] = []
					self.conf_paper[c].append(p)

		sumpapersconf, sumauthorsconf = 0, 0
		conf_authors = dict()
		for conf in self.conf_paper:
			papers = self.conf_paper[conf]
			sumpapersconf += len(papers)
			for paper in papers:
				if paper in self.paper_author:
					authors = self.paper_author[paper]
					sumauthorsconf += len(authors)

		print ("#confs  ", len(self.conf_paper))
		print ("#papers ", sumpapersconf,  "#papers per conf ", sumpapersconf / len(self.conf_paper))
		print ("#authors", sumauthorsconf, "#authors per conf", sumauthorsconf / len(self.conf_paper))


	def generate_random_aca(self):
		for conf in self.conf_paper:
			self.conf_authorlist[conf] = []
			for paper in self.conf_paper[conf]:
				if paper not in self.paper_author: continue
				for author in self.paper_author[paper]:
					self.conf_authorlist[conf].append(author)
					if author not in self.author_conflist:
						self.author_conflist[author] = []
					self.author_conflist[author].append(conf)

		print ("author-conf list done")

		con_list=[]
		author_list=[]
		for conf in self.conf_authorlist:
			for a in self.conf_authorlist[conf]:
				con_list.append(int(conf))
				author_list.append(int(a))

		print("list construct done")

		hg = dgl.heterograph({
			('conference', 'ca', 'author'): (con_list,author_list),
			('author', 'ac', 'conference'): (author_list, con_list)})
		return hg,self.id_conf,self.id_author

def main():
	#参数设置
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	iterations=4
	initial_lr = 0.025

	#构图
	mpg = MetaPathGenerator()
	mpg.read_data(dirpath)
	hg, id_conf, id_author=mpg.generate_random_aca()
	print(hg)
	nid2word={"conference":id_conf,
			 "author":id_author}
	metapath = ['ca', 'ac'] * 100
    #采用和example文件训练同样的参数，为了输出label对应的embedding方便test所以传入nid2word
	model = Metapath2vec(hg, 128, metapath, 7,node_repeat=1000,nid2word=nid2word)
	model.initParameters()
	model = model.to(device)
	print("initial ok")
	dataloader = model.loader(batch_size=50, num_workers=8)
	print("step2")
	optimizer = optim.SparseAdam(list(model.parameters()), lr=initial_lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

	logger.info("Total Embedding size:" + str(model.word_count))
	logger.info("real Embedding size:" + str(len(model.word_frequency)))
	logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated()))

	for iteration in range(iterations):
		logger.info('>' * 100)
		logger.info('epoch: {}'.format(iteration))
		print("\n\n\nIteration: " + str(iteration + 1))

		running_loss = 0.0
		for i, batched_data in enumerate(tqdm(dataloader)):
			if len(batched_data[0]) > 1:
				pos_u = batched_data[0].to(device)
				pos_v = batched_data[1].to(device)
				neg_v = batched_data[2].to(device)

				scheduler.step()
				optimizer.zero_grad()
				loss = model.forward(pos_u, pos_v, neg_v)
				loss.backward()
				optimizer.step()

				running_loss = running_loss * 0.9 + loss.item() * 0.1
				if i > 0 and i % 500 == 0:
					now = localtime()
					now_time = strftime("%Y-%m-%d %H:%M:%S", now)
					logger.info('time:{} loss: {:.4f}'.format(now_time, running_loss))
					print(" Loss: " + str(running_loss))

		save_embedding(model,model.id2word,"out",iteration)
	print("training is end")


def save_embedding(model,id2word, file_name, num):
	embedding = model.u_embeddings.weight.cpu().data.numpy()
	with open(file_name + "/" + "my"+ file_name + str(num) +strftime("%y%m%d-%H%M", localtime()) +".txt", 'w') as f:
		f.write('%d %d\n' % (len(id2word), model.emb_dimension))
		for wid, w in id2word.items():
			e = ' '.join(map(lambda x: str(x), embedding[wid]))
			f.write('%s %s\n' % (w, e))

if __name__ == "__main__":
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(logging.StreamHandler(sys.stdout))
	log_file = '{}-{}-{}.log'.format("dgl_meta", "aminer", strftime("%y%m%d-%H%M", localtime()))
	logger.addHandler(logging.FileHandler("log/" + log_file))
	main()