import torch
import numpy
from library.network.batch_processing.batching import BatchProcessor


class OptimizedBatchProcessor(BatchProcessor):
    def __init__(self, tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path):
        super().__init__(tensors_dir, language, authors_size, vocab_size, batch_size, timesteps, truth_file_path)
        self.current_author = 1

    def get_index(self):
        if self.authors_usage[self.current_author]+1 == self.authors_max[self.current_author]:
            self.current_author += 1

        print(self.current_author, self.authors_usage[self.current_author], self.authors_max[self.current_author])
        return self.current_author

    def add_new_index(self, index):
        self.authors_order.append(index)
        self.authors_usage[index] += 1
        if self.authors_usage[index] + self.batch_size == self.authors_max[index] and index == self.authors_size:
            self.has_next_batch = False
        elif self.authors_usage[index] == self.authors_max[index] and index != self.max_index:
            self.authors_usage[index] = 0

    def get_results(self):
        self.batches = []
        self.authors_order = []
        self.labels = numpy.zeros((self.batch_size, self.vocab_size), dtype=int)
        self.process()
        return self.batches, torch.tensor(self.convert_to_one_number(self.labels)), self.authors_order
