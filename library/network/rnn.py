import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from library.preprocessing.to_tensor.alphabets.en_alphabet import alphabet as en

hidden_size = 1024
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30
learning_rate = 0.002
authors_size = 2
vocab_size = len(en)

rep_tensor = torch.load('../../data/tensors/known/EN002/EN002.pt')
rep_tensor = rep_tensor.type(torch.FloatTensor)
num_batches = rep_tensor.shape[1] // timesteps

print(rep_tensor.shape)


class TextGenerator(nn.Module):

    def __init__(self, authors_size, vocab_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(614400, authors_size)

    def forward(self, x, h):
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1) * out.size(2))
        out = self.linear(out)
        return out, (h, c)


model = TextGenerator(authors_size, vocab_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))

    print("states[0]:", states[0].shape)
    print("states[1]:", states[1].shape)

    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        inputs = rep_tensor[:, i:i + timesteps]
        print(inputs.shape)
        target = torch.tensor([1, 0])
        outputs, _ = model(inputs, states)
        print("loss_fn outputs:", outputs.shape, "loss_fn targets:", target.reshape(-1).shape)
        print(outputs)
        print(target)
        print(outputs.shape)
        print(target.shape)
        loss = loss_fn(outputs, target.reshape(-1))

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // timesteps
        if step % 1 == 0:
            print('Epoch [{}/{}], Loss: {:4f}'.format(epoch + 1, num_epochs, loss.item()))
