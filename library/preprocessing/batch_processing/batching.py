import torch


class BatchProcessor:
    input_tensor = None
    batch_size = None
    batch_tensor = None
    batches = []

    def __init__(self, batch_size=20):
        self.input_tensor = None
        self.batch_size = batch_size

    def set_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.batches = []
        self.process()

    def process(self):
        batches_num = len(self.input_tensor) // self.batch_size
        for x in range(self.batch_size):
            if(len(self.input_tensor[x * batches_num:(x + 1) * batches_num]) >= batches_num):
                self.batches.append(self.input_tensor[x * batches_num:(x + 1) * batches_num])

    def get_batches(self):
        return self.batches
