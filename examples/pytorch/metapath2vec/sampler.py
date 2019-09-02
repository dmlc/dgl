import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import random
import time

Metapath = "Conference-Paper-Author-Paper-Conference"
num_walks_per_node = 1000
walk_length = 100

#construct mapping from text, could be changed to DGL later
def construct_id_dict():
    id_to_paper = {}
    id_to_author = {}
    id_to_conf = {}
    f_3 = open(".../id_author.txt", encoding="ISO-8859-1")
    f_4 = open(".../id_conf.txt", encoding="ISO-8859-1")
    f_5 = open(".../paper.txt", encoding="ISO-8859-1")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.split('\t')
        identity = int(z[0])
        id_to_author[identity] = z[1].strip("\n")
    while True:
        w = f_4.readline()
        if not w:
            break;
        w = w.split('\t')
        identity = int(w[0])
        id_to_conf[identity] = w[1].strip("\n")
    while True:
        v = f_5.readline()
        if not v:
            break;
        v = v.split(' ')
        identity = int(v[0])
        paper_name = ""
        for s in range(5, len(v)):
            paper_name += v[s]
        paper_name = 'p' + paper_name
        id_to_paper[identity] = paper_name.strip('\n')
    f_3.close()
    f_4.close()
    f_5.close()
    return id_to_paper, id_to_author, id_to_conf

#construct mapping from text, could be changed to DGL later
def construct_types_mappings():
    paper_to_author = {}
    author_to_paper = {}
    paper_to_conf = {}
    conf_to_paper = {}
    f_1 = open(".../paper_author.txt", "r")
    f_2 = open(".../paper_conf.txt", "r")
    for x in f_1:
        x = x.split('\t')
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        if x[0] in paper_to_author:
            paper_to_author[x[0]].append(x[1])
        else:
            paper_to_author[x[0]] = []
            paper_to_author[x[0]].append(x[1])
        if x[1] in author_to_paper:
            author_to_paper[x[1]].append(x[0])
        else:
            author_to_paper[x[1]] = []
            author_to_paper[x[1]].append(x[0])
    for y in f_2:
        y = y.split('\t')
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        if y[0] in paper_to_conf:
            paper_to_conf[y[0]].append(y[1])
        else:
            paper_to_conf[y[0]] = []
            paper_to_conf[y[0]].append(y[1])
        if y[1] in conf_to_paper:
            conf_to_paper[y[1]].append(y[0])
        else:
            conf_to_paper[y[1]] = []
            conf_to_paper[y[1]].append(y[0])
    f_1.close()
    f_2.close()
    return paper_to_author, author_to_paper, paper_to_conf, conf_to_paper

#"conference - paper - Author - paper - conference" metapath sampling
def generate_metapath():
    output_path = open(".../output_path.txt", "w")
    id_to_paper, id_to_author, id_to_conf = construct_id_dict()
    paper_to_author, author_to_paper, paper_to_conf, conf_to_paper = construct_types_mappings()
    count = 0
    #loop all conferences
    for conf_id in conf_to_paper.keys():
        start_time = time.time()
        print("sampling" + str(count))
        conf = id_to_conf[conf_id]
        conf0 = conf
        #for each conference, simulate num_walks_per_node walks
        for i in range(num_walks_per_node):
            outline = conf0
            # each walk with length walk_length
            for j in range(walk_length):
                # C - P
                paper_list_1 = conf_to_paper[conf_id]
                # check whether the paper nodes link to any author node
                connections_1 = False
                available_paper_1 = []
                for k in range(len(paper_list_1)):
                    if paper_list_1[k] in paper_to_author:
                        available_paper_1.append(paper_list_1[k])
                num_p_1 = len(available_paper_1)
                if num_p_1 != 0:
                    connections_1 = True
                    paper_1_index = random.randrange(num_p_1)
                    #paper_id_1 = paper_list_1[paper_1_index]
                    paper_id_1 = available_paper_1[paper_1_index]
                    paper_1 = id_to_paper[paper_id_1]
                    outline += " " + paper_1
                else:
                    break
                # C - P - A
                author_list = paper_to_author[paper_id_1]
                num_a = len(author_list)
                # No need to check
                author_index = random.randrange(num_a)
                author_id = author_list[author_index]
                author = id_to_author[author_id]
                outline += " " + author
                # C - P - A - P
                paper_list_2 = author_to_paper[author_id]
                #check whether paper node links to any conference node
                connections_2 = False
                available_paper_2 = []
                for m in range(len(paper_list_2)):
                    if paper_list_2[m] in paper_to_conf:
                        available_paper_2.append(paper_list_2[m])
                num_p_2 = len(available_paper_2)
                if num_p_2 != 0:
                    connections_2 = True
                    paper_2_index = random.randrange(num_p_2)
                    paper_id_2 = available_paper_2[paper_2_index]
                    paper_2 = id_to_paper[paper_id_2]
                    outline += " " + paper_2
                else:
                    break
                # C - P - A - P - C
                conf_list = paper_to_conf[paper_id_2]
                num_c = len(conf_list)
                conf_index = random.randrange(num_c)
                conf_id = conf_list[conf_index]
                conf = id_to_conf[conf_id]
                outline += " " + conf
            if connections_1 and connections_2:
                output_path.write(outline + "\n")
            else:
                break
            # Note that the original mapping text has type indicator in front of each node just like "cVLDB"
            # So the sampling sequence looks like "cconference ppaper aauthor ppaper cconference"
        count += 1
        print("--- %s seconds ---" % (time.time() - start_time))
    output_path.close()


if __name__ == "__main__":
    generate_metapath()
