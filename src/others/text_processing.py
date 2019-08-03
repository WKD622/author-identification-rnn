import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        rep_tensor = torch.LongTensor(tokens)
        index = 0

        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1

        num_batches = rep_tensor.shape[0] // batch_size
        rep_tensor = rep_tensor[:num_batches * batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor
