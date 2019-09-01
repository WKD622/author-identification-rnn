import torch

class BatchProcessor:
    input_tensor = None
    batch_size = None
    batch_tensor = None
    batches = []

    def __init__(self):
        self.input_tensor = None
        self.batch_size = None

    def set_params(self, input_tensor: torch.LongTensor, batch_size=20):
        self.input_tensor = input_tensor
        self.batch_size = batch_size
        self.process()

    def process(self):
        batches_num = self.input_tensor.shape[0] // self.batch_size
        self.batch_tensor = self.input_tensor[:self.batch_size*batches_num]
        batch_size = self.batch_size
        for x in range(batches_num):
            self.batches.append(self.batch_tensor[x * batch_size:(x+1) * batch_size])


    def get_batches(self):
        return self.batches


# t = torch.LongTensor(140)
# test = BatchProcessor(t)
# print(test.get_batches())



