import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        rep_tensor = torch.LongTensor(tokens)
        index = 0

        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1

        num_batches = rep_tensor.shape[0] // batch_size
        rep_tensor = rep_tensor[:num_batches * batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor


embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30
learning_rate = 0.002

corpus = TextProcess()
rep_tensor = corpus.get_data('../../data/alice', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = rep_tensor.shape[1] // timesteps


# class TextGenerator(nn.Module):
#
#     def __init__(self, authors_size, vocab_size, hidden_size, num_layers):
#         super(TextGenerator, self).__init__()
#         self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, authors_size)
#
#     def forward(self, x, h):
#         out, (h, c) = self.lstm(x, h)
#         out = out.reshape(out.size(0) * out.size(1), out.size(2))
#         out = self.linear(out)
#         return out, (h, c)

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


model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))
    print(states)

    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        inputs = rep_tensor[:, i:i + timesteps]
        targets = rep_tensor[:, (i + 1):(i + 1) + timesteps]
        outputs, _ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // timesteps
        if step % 1 == 0:
            print('Epoch [{}/{}], Loss: {:4f}'.format(epoch + 1, num_epochs, loss.item()))

with torch.no_grad():
    with open('results.txt', 'w') as f:
        state = (torch.zeros(num_layers, 1, hidden_size),
                 torch.zeros(num_layers, 1, hidden_size))

        input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            output, _ = model(input, state)
            print(output.shape)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            input.fill_(word_id)

            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, 500, 'results.txt'))
