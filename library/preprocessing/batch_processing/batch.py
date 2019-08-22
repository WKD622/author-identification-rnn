import torch


class BatchProcessor:
    input_tensor = None
    batch_size = None
    batch_tensor = None

    def __init__(self, input_tensor: torch.LongTensor, batch_size=20):
        self.input_tensor = input_tensor
        self.batch_size = batch_size
        self.process()

    def process(self):
        batches_num = self.input_tensor.shape[0] // self.batch_size
        self.batch_tensor = self.input_tensor[:self.batch_size * batches_num]
        self.batch_tensor = self.batch_tensor.view(batches_num, -1)

    def get_batches(self):
        return self.batch_tensor

# t = torch.LongTensor(140)
# test = BatchProcessor(t)
# print(test.get_batches())
