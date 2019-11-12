import torch
from random import randint, choice
import numpy


class BatchProcessor:
    def __init__(self, tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path=''):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.authors_size = authors_size
        self.language = language
        self.vocab_size = vocab_size
        self.tensors_dir = tensors_dir
        self.authors_usage = numpy.zeros(authors_size+1, dtype=int)
        self.forbidden_index = []
        self.batches = []
        self.authors_order = []
        self.max_length = None
        self.has_next_batch = True
        self.max_index = 0
        self.authors_max = numpy.zeros(authors_size+1, dtype=int)
        self.eligible_authors = []
        self.truth_file_path = truth_file_path
        self.parse_truth()
        self.set_max_length()

    def parse_truth(self):
        return

    def set_max_length(self):
        max_size = len(self.load_tensor(1))
        for i in range(1, self.authors_size+1):
            size = len(self.load_tensor(i))
            if max_size < size:
                max_size = size
                max_index = i
            self.authors_max[i] = size // self.timesteps - 1

        self.max_length = max_size // self.timesteps - 1
        self.max_index = max_index

    def set_tensors(self):
        batches = numpy.zeros((self.batch_size, self.timesteps, self.vocab_size), dtype=int)
        for x in range(0, self.batch_size):
            index = self.get_index()
            self.add_new_index(index)
            input_tensor = self.load_tensor(index)
            # if index == self.max_index:
            #     print(index, self.authors_usage[index], self.authors_max[index])
            #     print(self.has_next_batch)
            delta = self.authors_usage[index]*self.timesteps
            for y in range(0, self.timesteps):
                batches[x][y] = input_tensor[delta + y]
        self.batches = torch.from_numpy(batches)

    def get_index(self):
        index = randint(1, self.authors_size)
        if index in self.forbidden_index:
            while index in self.forbidden_index:
                index = randint(1, self.authors_size)
        return index

    def add_new_index(self, index):
        self.authors_order.append(index-1)
        self.authors_usage[index] += 1
        if self.authors_usage[index] == self.max_length and index == self.max_index:
            self.forbidden_index.append(index)
            self.has_next_batch = False
        elif self.authors_usage[index] == self.authors_max[index] and index != self.max_index:
            self.authors_usage[index] = 0

    def load_tensor(self, index):
        idx = str(index).zfill(3)
        dir_name = self.language + idx
        path = self.tensors_dir + dir_name + '/' + dir_name + '.pt'
        tensor = torch.load(path)
        return tensor

    def process(self):
        self.set_tensors()

    def get_batch(self):
        return self.batches

    def get_labels(self):
        return self.authors_order

    def new_epoch(self):
        self.forbidden_index = []
        self.authors_usage = numpy.zeros(self.authors_size + 1, dtype=int)
        self.has_next_batch = True

    def next_batch(self):
        return self.has_next_batch

    def get_results(self):
        self.batches = []
        self.authors_order = []
        self.process()
        return self.batches, self.authors_order