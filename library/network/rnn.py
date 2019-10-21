import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import sys 

sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')

from library.network.batch_processing.batching import BatchProcessor
from library.preprocessing.to_tensor.alphabets.en_alphabet import alphabet as en

hidden_size = 100
num_layers = int(sys.argv[1])
num_epochs = 5
batch_size = 40
timesteps = 50
learning_rate = 0.004
authors_size = 100
vocab_size = len(en)



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
# model.load_state_dict(torch.load(save_path))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

j = 0
while True:
    batch_processor = BatchProcessor(batch_size=batch_size, authors_size=authors_size, timesteps=timesteps)
    i = 0
    j += 1
    losses = []
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))

        i += 1
        batch_processor.new_epoch()
        losses = []
        while batch_processor.next_batch():
            batches, labels = batch_processor.get_results()
            batches = batches.type(torch.FloatTensor)
            target = torch.tensor(labels)
            outputs, _ = model(batches, states)
            loss = loss_fn(outputs, target)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // timesteps

    if j % 100 == 0:
        loss_avg = sum(losses) / len(losses)
        save_path = 'results'+str(num_layers)+'/'+str(j)+'loss:'+str(loss_avg)
    torch.save(model.state_dict(), save_path)
