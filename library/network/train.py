import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import sys
from .model import TextGenerator

sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')

from library.network.batch_processing.batching import BatchProcessor
from library.preprocessing.to_tensor.alphabets.en_alphabet import alphabet as en


class Train:

    def __init__(self, hidden_size, num_layers, num_epochs, batch_size, timesteps, learning_rate, authors_size, path):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.authors_size = authors_size
        self.vocab_size = len(en)
        self.path = path

        self.model = TextGenerator(self.authors_size, self.vocab_size, self.hidden_size, self.num_layers,
                                   self.timesteps)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        outer_counter = 0
        while True:
            batch_processor = BatchProcessor(batch_size=self.batch_size, authors_size=self.authors_size,
                                             timesteps=self.timesteps)
            outer_counter += 1
            epochs_counter = 0
            losses = []
            for epoch in range(self.num_epochs):
                states = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                          torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

                epochs_counter += 1
                batch_processor.new_epoch()
                losses = []
                while batch_processor.next_batch():
                    batches, labels = batch_processor.get_results()
                    batches = batches.type(torch.FloatTensor)
                    target = torch.tensor(labels)
                    outputs, _ = self.model(batches, states)
                    loss = self.loss_fn(outputs, target)
                    losses.append(loss.item())

                    self.model.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            self.output(outer_counter, losses)

    def give_output(self, outer_counter, losses):
        pass

    def save_model(self, outer_counter, losses):
        if outer_counter % 100 == 0:
            loss_avg = sum(losses) / len(losses)
            save_path = self.path + '/' + str(outer_counter) + 'loss:' + str(loss_avg)
            torch.save(self.model.state_dict(), save_path)

    def output(self, outer_counter, losses):
        self.save_model(outer_counter, losses)
        self.give_output(outer_counter, losses)

    def accuracy(self, labels, target):
        pass
