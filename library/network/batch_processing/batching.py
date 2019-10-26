import torch
from random import randint
import numpy


class BatchProcessor:
    authors_usage = []
    batch_size = None
    batch_tensor = None
    batches = []
    authors_size = None
    timesteps = None
    language = None
    vocab_size = None
    authors_order = []
    labels = []
    max_length = None
    has_next_batch = True

    def __init__(self, tensors_dir, batch_size=40, authors_size=100, timesteps=30, language='EN', vocab_size=40):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.authors_size = authors_size
        self.language = language
        self.vocab_size = vocab_size
        self.authors_usage = numpy.zeros(authors_size + 1, dtype=int)
        self.set_max_length()
        self.tensors_dir = tensors_dir

    def set_max_length(self):
        min_size = len(self.load_tensor(1))
        for i in range(1, self.authors_size):
            size = len(self.load_tensor(i))
            min_size = min(size, min_size)

        self.max_length = min_size // self.timesteps - 1

    def set_tensors(self):
        batches = numpy.zeros((self.batch_size, self.timesteps, self.vocab_size), dtype=int)
        forbidden_index = []
        for x in range(0, self.batch_size):
            index = randint(1, self.authors_size)
            if index in forbidden_index:
                while index in forbidden_index:
                    index = randint(1, self.authors_size)
            self.authors_order.append(index - 1)
            self.authors_usage[index] += 1
            if self.authors_usage[index] == self.max_length:
                forbidden_index.append(index)
                self.has_next_batch = False
            input_tensor = self.load_tensor(index)
            delta = self.authors_usage[index] * self.timesteps
            for y in range(0, self.timesteps):
                batches[x][y] = input_tensor[delta + y]
        self.batches = torch.from_numpy(batches)

    def load_tensor(self, index):
        idx = str(index).zfill(3)
        dir_name = self.language + idx
        path = self.tensors_dir + dir_name + '/' + dir_name + '.pt'
        tensor = torch.load(path)
        return tensor

    # def create_labels(self):
    #     labels = numpy.zeros((self.batch_size, self.authors_size), dtype=int)
    #     for i in range(0, self.batch_size):
    #         author_index = self.authors_order[i]
    #         labels[i][author_index] = 1
    #
    #     self.labels = torch.from_numpy(labels)

    def process(self):
        self.set_tensors()
        # self.create_labels()

    def get_batch(self):
        return self.batches

    def get_labels(self):
        return self.authors_order

    def new_epoch(self):
        self.authors_usage = numpy.zeros(self.authors_size + 1, dtype=int)
        self.has_next_batch = True

    def next_batch(self):
        return self.has_next_batch

    def get_results(self):
        self.batches = []
        self.authors_order = []
        # if self.has_next_batch:
        self.process()
        return self.batches, self.authors_order
        # else:
#             throw error


# t = torch.LongTensor(140)
# test = BatchProcessor(t)
# print(test.get_batches())

# x = BatchProcessor()
# # x.set_tensors()

# x = BatchProcessor()
# while(x.next_batch()):
#     x.get_results()
