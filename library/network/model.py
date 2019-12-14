import torch.nn as nn


class MultiHeadedRnn(nn.Module):

    def __init__(self, batch_size, authors_size, vocab_size, hidden_size, num_layers, timesteps):
        super(MultiHeadedRnn, self).__init__()
        self.authors_size = authors_size
        self.timesteps = timesteps
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for i in range(authors_size)])

    def forward(self, i, h):
        out, (h, c) = self.lstm(i, h)
        last_output = out[:, -1, :]
        tanh_output = self.tanh(last_output)
        heads_outputs = []
        for i in range(self.authors_size):
            heads_outputs.append(self.linears[i](tanh_output))
        return heads_outputs, h
