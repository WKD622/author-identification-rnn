import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

from library.network.batch_processing.batching import BatchProcessor
from library.preprocessing.to_tensor.alphabets.en_alphabet import alphabet as en

hidden_size = 100
num_layers = 2
num_epochs = 5
batch_size = 40
timesteps = 50
learning_rate = 0.004
authors_size = 100
vocab_size = len(en)
save_path = "result"


class TextGenerator(nn.Module):

    def __init__(self, authors_size, vocab_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(timesteps * hidden_size, authors_size)

    def forward(self, x, h):
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0), out.size(1) * out.size(2))
        out = self.linear(out)
        return out, (h, c)


model = TextGenerator(authors_size, vocab_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batch_processor = BatchProcessor(batch_size=batch_size, authors_size=authors_size, timesteps=timesteps)

while True:
    j = 0
    i = 0
    j += 1
    print(j)
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))

        i += 1
        batch_processor.new_epoch()
        while batch_processor.next_batch():
            batches = batch_processor.get_results()[0]
            labels = batch_processor.get_results()[1]
            batches = batches.type(torch.FloatTensor)
            target = torch.tensor(labels)
            outputs, _ = model(batches, states)
            loss = loss_fn(outputs, target)

            model.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // timesteps

        print('Epoch [{}/{}], Loss: {:4f}'.format(epoch + 1, num_epochs, loss.item()))
    torch.save(model.state_dict(), save_path)
