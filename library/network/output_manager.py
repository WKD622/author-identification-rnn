import datetime
import os

import torch

from library.helpers.files.files_operations import append_to_file, create_directory, create_file


class OutputManager:
    EPOCH = 'epoch'
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    MODEL = 'model'
    RESULTS_FILENAME = 'results.csv'
    NETWORK_INFO_FILENAME = 'network_info.txt'
    SEPARATOR = ','
    TIME_PASSED = 'time'
    HEADLINE = MODEL + SEPARATOR + EPOCH + SEPARATOR + LOSS + SEPARATOR + ACCURACY + SEPARATOR + TIME_PASSED + '\n'
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
        self.min_loss = 1000
        self.max_accuracy = 0.0

        self.initialize_files()
        self.outputs_counter = 1

    def next_output(self, model, losses, accuracy, epoch_number, time_passed):
        formatted_time_passed = str(datetime.timedelta(seconds=time_passed))
        loss_avg = sum(losses) / len(losses)
        if loss_avg <= self.min_loss or accuracy >= self.max_accuracy:
            self.save_model(model)
        self.console_output(loss_avg, accuracy, epoch_number, time_passed=formatted_time_passed)
        self.file_output(loss_avg, accuracy, epoch_number, time_passed=formatted_time_passed)
        self.update_max_loss_and_accuracy(loss=loss_avg, accuracy=accuracy)
        self.outputs_counter += 1

    def console_output(self, loss_avg, accuracy, epoch_number, time_passed):
        print(str(self.outputs_counter) +
              ' ' + self.EPOCH + ': ' + str(epoch_number) +
              ' ' + self.LOSS + ': ' + str(loss_avg) +
              ' ' + self.ACCURACY + ': ' + str(accuracy) +
              ' ' + self.TIME_PASSED + ': ' + str(time_passed))

    def save_model(self, model):
        save_path = os.path.join(self.models_path, str(self.outputs_counter))
        torch.save(model.state_dict(), save_path)

    def file_output(self, loss_avg, accuracy, epoch_number, time_passed):
        file_path = os.path.join(self.results_path, self.RESULTS_FILENAME)
        append_to_file(file_path,
                       str(self.outputs_counter) +
                       self.SEPARATOR + str(epoch_number) +
                       self.SEPARATOR + str(loss_avg) +
                       self.SEPARATOR + str(accuracy) +
                       self.SEPARATOR + str(time_passed) + '\n')

    def initialize_files(self):
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

    def update_max_loss_and_accuracy(self, loss, accuracy):
        if loss < self.min_loss:
            self.min_loss = loss
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
