import torch
from random import randint, choice
import numpy
from library.network.batch_processing.batching import BatchProcessor


class EvaluationBatchProcessor(BatchProcessor):
    def __init__(self, tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path):
        super().__init__(tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path)

    def parse_truth(self):
        with open(self.truth_file_path) as truth_file:
            truth_array = [next(truth_file) for x in range(self.authors_size)]
        for line in truth_array:
            if line != '' and line.split()[1] == 'Y':
                label = line.split()[0]
                self.eligible_authors.append(int(label[-3:]))

    def set_max_length(self):
        max_size = len(self.load_tensor(1))
        max_index = 1
        for i in self.eligible_authors:
            size = len(self.load_tensor(i))
            if max_size < size:
                max_size = size
                max_index = i
            self.authors_max[i] = size - 2 * self.timesteps - 2

        self.max_length = max_size - 2 * self.timesteps - 2
        self.max_index = max_index

    def get_index(self):
        index = choice(self.eligible_authors)
        if index in self.forbidden_index or self.is_not_a_file(index):
            while index in self.forbidden_index and self.is_not_a_file(index):
                index = choice(self.eligible_authors)

        return index

    def get_results(self):
        self.batches = []
        self.authors_order = []
        self.labels = numpy.zeros((self.batch_size, self.vocab_size), dtype=int)
        self.process()
        return self.batches, torch.tensor(self.convert_to_one_number(self.labels)), self.authors_order
