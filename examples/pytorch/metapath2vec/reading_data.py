import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(12345)

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count, care_type):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        #index2type = {}
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            #index2type[wid] = w[0]
            self.word_frequency[wid] = c
            wid += 1
        #type2indices = {}
        #all_types = set(index2type.values())
        #for node_type in all_types:
        #    type2indices[node_type] = []

        #for node_index, node_type in index2type.items():
        #    type2indices[node_type].append(node_index)

        #for node_type in all_types:
        #    type2indices[node_type] = np.array(type2indices[node_type])

        self.word_count = len(self.word2id)
        #self.index2type = index2type
        #self.type2indices = type2indices
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

        #all_types = set(self.index2type.values())
        #type2probs = {}
        #for node_type in all_types:
        #    indicies_for_a_type = self.type2indices[node_type]
        #    type2probs[node_type] = np.array(ratio[indicies_for_a_type])
        #    type2probs[node_type] = type2probs[node_type] / np.sum(type2probs[node_type])

        self.sampling_prob = ratio
        #self.type2probs = type2probs

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            #response = np.random.choice(
                #self.negatives, size=sizes)
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        #else:
            #node_type = self.index2type[target]
            #sampling_probs = self.type2probs[node_type]
            #sampling_candidates = self.type2indices[node_type]
            #negative_samples_indices = np.random.choice(len(sampling_candidates), size=sizes, replace=False,
                                                               #p=sampling_probs)
            #response = sampling_candidates[negative_samples_indices]
        return response


# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    #boundary = np.random.randint(1, self.window_size)
                    #return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            #enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]
                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v,5)))
                    return pair_catch


    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
