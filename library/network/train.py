import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from library.network.batch_processing.batching import BatchProcessor
from library.network.batch_processing.evaluation_batches import EvaluationBatchProcessor
from library.network.model import TextGenerator
from library.preprocessing.files.files_operations import create_file


class Train:

    def __init__(self, hidden_size, num_layers, num_epochs, batch_size, timesteps, learning_rate, authors_size,
                 save_path, tensors_path, language, vocab_size):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.authors_size = authors_size
        self.vocab_size = vocab_size
        self.tensors_path = tensors_path
        self.language = language

        self.model = TextGenerator(self.authors_size,
                                   self.vocab_size,
                                   self.hidden_size,
                                   self.num_layers,
                                   self.timesteps)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.output_manager = OutputManager(save_path=save_path)

    def train(self):
        counter = 0
        while True:
            batch_processor = BatchProcessor(tensors_dir=self.tensors_path,
                                             batch_size=self.batch_size,
                                             authors_size=self.authors_size,
                                             timesteps=self.timesteps,
                                             language=self.language,
                                             vocab_size=self.vocab_size)
            losses = []
            counter += 1
            for epoch in range(self.num_epochs):
                states = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                          torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

                batch_processor.new_epoch()
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

            self.output_manager.next_output(model=self.model,
                                            losses=losses,
                                            accuracy=self.get_accuracy(),
                                            epoch_number=self.num_epochs * counter)

    def get_accuracy(self):
        return 1


class OutputManager:
    FILENAME = 'results'

    def __init__(self, save_path):
        self.save_path = save_path
        self.initialize_files()
        self.outputs_counter = 1

    def next_output(self, model, losses, accuracy, epoch_number):
        loss_avg = sum(losses) / len(losses)
        self.save_model(model)
        self.console_output(losses, accuracy, epoch_number)
        self.file_output(losses, accuracy, epoch_number)
        self.outputs_counter += 1

    def console_output(self, losses, accuracy, epoch_number):
        print(str(self.outputs_counter) +
              ' epoch: ' + str(epoch_number) +
              ' loss: ' + losses +
              ' accuracy: ' + accuracy)

    def save_model(self, model):
        save_path = self.save_path + '/' + str(self.outputs_counter)
        torch.save(model.state_dict(), save_path)

    def file_output(self, losses, accuracy, epoch_number):
        with open(self.save_path + '/' + self.FILENAME, 'a') as results:
            results.write(str(self.outputs_counter) +
                          ' epoch: ' + str(epoch_number) +
                          ' loss: ' + losses +
                          ' accuracy: ' + accuracy)
            results.close()

    def initialize_files(self):
        create_file(self.save_path, self.FILENAME)
