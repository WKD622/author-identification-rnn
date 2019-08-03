from torch import nn


class TextGenerator(nn.Module):

    def __init__(self, vocab_size, emded_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, emded_size)
        self.lstm = nn.LSTM(emded_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)
