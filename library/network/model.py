import torch.nn as nn


class TextGenerator(nn.Module):

    def __init__(self, authors_size, vocab_size, hidden_size, num_layers, timesteps):
        super(TextGenerator, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(timesteps * hidden_size, authors_size)

    def forward(self, x, h):
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0), out.size(1) * out.size(2))
        out = self.linear(out)
        return out, (h, c)