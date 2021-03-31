
import configparser
import os
import numpy as np
import torch as th


class Config(object):
    def __init__(self, file_path, model, dataset, gpu):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0 :
            if th.cuda.is_available():
                self.device = th.device('cuda', int(gpu))
            else:
                print("cuda is not available")

        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.model = model
        self.dataset = dataset
        self.path = {'output_modelfold': './output/model/',
                     'input_fold': './dataset/'+self.dataset+'/',
                     'temp_fold': './output/temp/'+self.model+'/'}

        self.lr = conf.getfloat("HetGNN", "learning_rate")

        self.weight_decay = conf.getfloat("HetGNN", "weight_decay")
        # self.dropout = conf.getfloat("CompGCN", "dropout")
        self.max_epoch = conf.getint("HetGNN", "max_epoch")
        self.dim = conf.getint("HetGNN", "dim")
        self.batch_size = conf.getint("HetGNN", "batch_size")
        self.window_size = conf.getint("HetGNN", "window_size")
        self.num_workers = conf.getint("HetGNN", "num_workers")
        self.batches_per_epoch = conf.getint("HetGNN", "batches_per_epoch")

        self.rw_length = conf.getint("HetGNN", "rw_length")
        self.rw_walks = conf.getint("HetGNN", "rw_walks")
        self.rwr_prob = conf.getfloat("HetGNN", "rwr_prob")