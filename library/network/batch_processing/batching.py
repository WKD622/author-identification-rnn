import torch
from random import randint
import numpy

tensors_dir = '../../../data/tensors/known/'


class BatchProcessor:
    input_tensors = []
    batch_size = None
    batch_tensor = None
    batches = []
    authors_size = None
    timesteps = None
    language = None
    vocab_size = None
    authors_order = []
    labels = []

    def __init__(self, batch_size=20, authors_size=100, timesteps=30, language='EN', vocab_size=40):
        self.batch_size = batch_size
        self.authors_size = authors_size
        self.timesteps = timesteps
        self.language = language
        self.vocab_size = vocab_size
        self.process()

    def set_tensors(self):
        batches = numpy.zeros((self.batch_size, self.timesteps, self.vocab_size), dtype=int)
        for x in range(0, self.batch_size):
            index = randint(1, self.authors_size)
            self.authors_order.append(index)
            input_tensor = self.load_tensor(index)
            for y in range(0, self.timesteps):
                batches[x][y] = input_tensor[y]
        self.batches = torch.from_numpy(batches)

    def load_tensor(self, index):
        idx = str(index).zfill(3)
        dir_name = self.language + idx
        path = tensors_dir + dir_name + '/' + dir_name + '.pt'
        tensor = torch.load(path)
        return tensor

    def create_labels(self):
        labels = numpy.zeros((self.batch_size, self.authors_size), dtype=int)
        for i in range(0, self.batch_size):
            author_index = self.authors_order[i]
            labels[i][author_index] = 1

        self.labels = torch.from_numpy(labels)

    def process(self):
        self.set_tensors()
        self.create_labels()

    def get_batches(self):
        return self.batches

    def get_labels(self):
        return self.labels

    def get_results(self):
        return self.batches, self.labels


# t = torch.LongTensor(140)
# test = BatchProcessor(t)
# print(test.get_batches())

x = BatchProcessor()
# x.set_tensors()
