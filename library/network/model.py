import torch.nn as nn


class MultiHeadedRnn(nn.Module):

    def __init__(self, batch_size, authors_size, vocab_size, hidden_size, num_layers, timesteps):
        super(MultiHeadedRnn, self).__init__()
        self.batch_size = batch_size
        self.authors_size = authors_size
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)
        self.linears = [nn.Linear(timesteps * hidden_size, vocab_size) for i in range(authors_size)]

    def forward(self, i, h):
        out, (h, c) = self.lstm(i, h)
        out = out.reshape(out.size(0), out.size(1) * out.size(2))
        outs = []
        for i in range(self.authors_size):
            outs.append(self.linears[i](out))
        return outs, (h, c)
