import torch
import torch.nn as nn
from torchviz import make_dot


class MultiHeadedRnn(nn.Module):

    def __init__(self, batch_size, authors_size, vocab_size, hidden_size, num_layers, timesteps):
        super(MultiHeadedRnn, self).__init__()
        self.batch_size = batch_size
        self.authors_size = authors_size
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)
        self.linears = [nn.Linear(hidden_size, vocab_size) for i in range(authors_size)]

    def forward(self, i, h):
        out, (h, c) = self.lstm(i, h)
        out_ = torch.zeros(self.batch_size, self.hidden_size)
        for q in range(self.batch_size):
            for e in range(self.hidden_size):
                out_[q][e] = out[q][self.timesteps - 1][e]
        outs = []
        for i in range(self.authors_size):
            outs.append(self.linears[i](out_))
        return outs, (h, c)

    # counter = 0
    # for author in i:
    #     for vector in author:
    #         print(vector)
    #         for elem in vector:
    #             if elem == 1:
    #                 counter += 1
    # print(50*20 - counter)
    # print(i[0][0])
    # def forward(self, i, h):
    #     out, (h, c) = self.lstm(i, h)
    #     print(out.shape)
    #     out = out.reshape(out.size(0), out.size(1) * out.size(2))
    #     outs = []
    #     for i in range(self.authors_size):
    #         outs.append(self.linears[i](out))
    #     return outs, (h, c)
