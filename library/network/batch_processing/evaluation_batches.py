import torch
from random import randint, choice
import numpy
from library.network.batch_processing.batching import BatchProcessor


class EvaluationBatchProcessor(BatchProcessor):
    def __init__(self, tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path):
        super().__init__(tensors_dir, language, authors_size, vocab_size, batch_size, timesteps)
        self.eligible_authors = []
        self.truth_file_path = truth_file_path
        self.parse_truth()

    def set_max_length(self):
        min_size = len(self.load_tensor(1))
        for i in range(1, self.authors_size):
            size = len(self.load_tensor(i))
            min_size = min(size, min_size)

        self.max_length = min_size // self.timesteps - 1

    def parse_truth(self):
        with open(self.truth_file_path) as truth_file:
            truth_array = [next(truth_file) for x in range(self.authors_size)]
        for line in truth_array:
            if line != '' and line.split()[1] == 'Y':
                label = line.split()[0]
                self.eligible_authors.append(int(label[-3:]))

    def get_index(self):
        index = choice(self.eligible_authors)
        if index in self.forbidden_index:
            while index in self.forbidden_index:
                index = choice(self.eligible_authors)

        return index

    def get_results(self):
        self.batches = []
        self.authors_order = []
        self.process()
        return self.batches, self.authors_order
