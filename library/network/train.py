import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import sys

sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')
from library.network.batch_processing.batching import BatchProcessor
from library.network.model import TextGenerator
from library.helpers.files.files_operations import (create_file, append_to_file, create_directory,
                                                    remove_directory, check_if_directory)


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
        self.output_manager = OutputManager(save_path=save_path,
                                            hidden_size=hidden_size,
                                            num_epochs=num_epochs,
                                            num_layers=num_layers,
                                            batch_size=batch_size,
                                            timesteps=timesteps,
                                            learning_rate=learning_rate,
                                            authors_size=authors_size,
                                            vocab_size=vocab_size)

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
        # batch_processor = EvaluationBatchProcessor(tensors_dir=self.tensors_path,
        #                                            batch_size=self.batch_size,
        #                                            authors_size=self.authors_size,
        #                                            timesteps=self.timesteps,
        #                                            language=self.language,
        #                                            vocab_size=self.vocab_size)
        # TODO
        return 1.1


class OutputManager:
    EPOCH = 'epoch'
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    MODEL = 'model'
    RESULTS_FILENAME = 'results.csv'
    NETWORK_INFO_FILENAME = 'network_info.txt'
    SEPARATOR = ','
    HEADLINE = MODEL + SEPARATOR + EPOCH + SEPARATOR + LOSS + SEPARATOR + ACCURACY + '\n'
    MODELS_FOLDER_NAME = 'models'

    def __init__(self, save_path, hidden_size, num_layers, num_epochs, batch_size, timesteps, learning_rate,
                 authors_size, vocab_size):
        self.models_path = os.path.join(save_path, self.MODELS_FOLDER_NAME)
        self.results_path = save_path

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.authors_size = authors_size
        self.vocab_size = vocab_size

        self.initialize_files()
        self.outputs_counter = 1

    def next_output(self, model, losses, accuracy, epoch_number):
        loss_avg = sum(losses) / len(losses)
        self.save_model(model)
        self.console_output(loss_avg, accuracy, epoch_number)
        self.file_output(loss_avg, accuracy, epoch_number)
        self.outputs_counter += 1

    def console_output(self, loss_avg, accuracy, epoch_number):
        print(str(self.outputs_counter) +
              ' ' + self.EPOCH + ': ' + str(epoch_number) +
              ' ' + self.LOSS + ': ' + str(loss_avg) +
              ' ' + self.ACCURACY + ': ' + str(accuracy))

    def save_model(self, model):
        save_path = os.path.join(self.models_path, str(self.outputs_counter))
        torch.save(model.state_dict(), save_path)

    def file_output(self, loss_avg, accuracy, epoch_number):
        file_path = os.path.join(self.results_path, self.RESULTS_FILENAME)
        append_to_file(file_path,
                       str(self.outputs_counter) +
                       self.SEPARATOR + str(epoch_number) +
                       self.SEPARATOR + str(loss_avg) +
                       self.SEPARATOR + str(accuracy) + '\n')

    def initialize_files(self):
        if check_if_directory(self.results_path):
            remove_directory(self.results_path)
        create_file(filename=self.RESULTS_FILENAME, path=self.results_path)
        create_file(filename=self.NETWORK_INFO_FILENAME, path=self.results_path)
        self.add_results_headline()
        self.add_network_info()
        create_directory(self.models_path)

    def add_results_headline(self):
        path = os.path.join(self.results_path, self.RESULTS_FILENAME)
        append_to_file(path, self.HEADLINE)

    def add_network_info(self):
        path = os.path.join(self.results_path, self.NETWORK_INFO_FILENAME)
        append_to_file(path,
                       'num_layers: ' + str(self.num_layers) + '\n' +
                       'hidden_size: ' + str(self.hidden_size) + '\n' +
                       'batch_size: ' + str(self.batch_size) + '\n' +
                       'timesteps: ' + str(self.timesteps) + '\n' +
                       'learning_rate: ' + str(self.learning_rate) + '\n' +
                       'num_epochs: ' + str(self.num_epochs) + '\n' +
                       'vocab_size: ' + str(self.vocab_size) + '\n' +
                       'authors_size: ' + str(self.authors_size) + '\n')
