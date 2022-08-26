import dgl


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