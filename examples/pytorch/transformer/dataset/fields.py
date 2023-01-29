class Vocab:
    def __init__(
        self, init_token=None, eos_token=None, pad_token=None, unk_token=None
    ):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab_lst = []
        self.vocab_dict = None

    def load(self, path):
        if self.init_token is not None:
            self.vocab_lst.append(self.init_token)
        if self.eos_token is not None:
            self.vocab_lst.append(self.eos_token)
        if self.pad_token is not None:
            self.vocab_lst.append(self.pad_token)
        if self.unk_token is not None:
            self.vocab_lst.append(self.unk_token)
        with open(path, "r", encoding="utf-8") as f:
            for token in f.readlines():
                token = token.strip()
                self.vocab_lst.append(token)
        self.vocab_dict = {v: k for k, v in enumerate(self.vocab_lst)}

    def __len__(self):
        return len(self.vocab_lst)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.vocab_dict:
                return self.vocab_dict[key]
            else:
                return self.vocab_dict[self.unk_token]
        else:
            return self.vocab_lst[key]


class Field:
    def __init__(self, vocab, preprocessing=None, postprocessing=None):
        self.vocab = vocab
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x

    def postprocess(self, x):
        if self.postprocessing is not None:
            return self.postprocessing(x)
        return x

    def numericalize(self, x):
        return [self.vocab[token] for token in x]

    def __call__(self, x):
        return self.postprocess(self.numericalize(self.preprocess(x)))
