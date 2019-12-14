import sys
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from library.helpers.files.files_operations import append_to_file, create_file
from library.network.output_manager import OutputManager

sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')
from library.network.batch_processing.batching import BatchProcessor
from library.network.batch_processing.batching_optimized import OptimizedBatchProcessor
from library.network.model import MultiHeadedRnn


class Train:

    def __init__(self, hidden_size, num_layers, num_epochs, batch_size, timesteps, learning_rate, authors_size,
                 save_path, training_tensors_path, testing_tensors_path, language, vocab_size):
        create_file('output.txt', '.', '')
        append_to_file('output.txt', 'start\n')

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.authors_size = authors_size
        self.vocab_size = vocab_size
        self.training_tensors_path = training_tensors_path
        self.testing_tensors_path = testing_tensors_path
        self.language = language
        self.save_path = save_path
        self.time_start = 0
        self.model = MultiHeadedRnn(self.batch_size,
                                    self.authors_size,
                                    self.vocab_size,
                                    self.hidden_size,
                                    self.num_layers,
                                    self.timesteps)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
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
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss(reduction='none')

    def train(self):
        self.time_start = time.time()
        counter = 0
        states = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        while True:
            batch_processor = BatchProcessor(tensors_dir=self.training_tensors_path,
                                             batch_size=self.batch_size,
                                             authors_size=self.authors_size,
                                             timesteps=self.timesteps,
                                             language=self.language,
                                             vocab_size=self.vocab_size)
            counter += 1
            batch_processor.new_epoch()
            torch.autograd.set_detect_anomaly(True)
            while batch_processor.next_batch():
                batches, target, authors_order = batch_processor.get_results()
                batches = batches.type(torch.FloatTensor)
                outputs, _ = self.model(batches, states)
                # self.print_input_sequence(batches, target, outputs)
                heads_to_train = self.get_heads_for_training(authors_order)
                loss = 0
                for head in heads_to_train:
                    # creating mask
                    mask = (torch.tensor(authors_order) == head + 1).float()

                    # calculating softmax
                    softmax = self.softmax(outputs[head])

                    # calculating loss which is a vector of same size as outputs[head]
                    vector = self.loss(softmax, target)

                    # then we equalize to 0 elements of vector we don't need
                    vector = vector * mask

                    # and finally...
                    loss = torch.add(torch.sum(vector) / torch.sum(mask), loss)

                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            accuracy, loss = self.get_accuracy(i=counter)
            self.output_manager.next_output(model=self.model,
                                            losses=[loss],
                                            accuracy=accuracy,
                                            epoch_number=counter,
                                            time_passed=time.time() - self.time_start)

    def get_heads_for_training(self, authors_order):
        heads = []
        for author in authors_order:
            heads.append(author - 1)
        return heads

    # def print_input_sequence(self, sequence, target, outputs):
    #     for index, author in enumerate(sequence):
    #         print('AUTHOR ' + str(index) + ') SEQUENCE: |', end='')
    #         for timestep in author:
    #             print(decode_letter(timestep).replace('\n', ' '), end='')
    #         target_ = decode_letter(class_to_one_hot(target)).replace('\n', ' ')
    #         print('|     TARGET: |' + decode_letter(class_to_one_hot(target)).replace('\n', ' ') + '|', end='  ')
    #         max_index = outputs[0][0].argmax(0)
    #         output = decode_letter(class_to_one_hot(max_index + 1)).replace('\n', ' ')
    #         print('OUTPUT: |' + decode_letter(class_to_one_hot(max_index + 1)).replace('\n', ' ') + '|', end='   ')
    #         if output == target_:
    #             print('MATCHED', end='      ')

    def get_accuracy(self, i):
        """
        Pushes all unknown texts on all heads and for each text finds head which has the smallest
        average cross entropy value.
        :param i:
        :return:
        """
        batch_processor = BatchProcessor(tensors_dir=self.testing_tensors_path,
                                         batch_size=self.batch_size,
                                         authors_size=self.authors_size,
                                         timesteps=self.timesteps,
                                         language=self.language,
                                         vocab_size=self.vocab_size)

        batch_processor.new_epoch()
        states = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

        average_cross_entropies = self.get_average_cross_entropies()
        testing_data_looses = self.initialize_testing_loss_struct()

        append_to_file('output.txt', 'average_cross_entropies\n' + str(average_cross_entropies))
        append_to_file('output.txt', '\n')

        while batch_processor.next_batch():
            # here we start using evaluation data
            batches, target, authors_order = batch_processor.get_results()
            batches = batches.type(torch.FloatTensor)
            outputs, _ = self.model(batches, states)

            # iterating through all heads
            for head in range(self.authors_size):
                # calculating cross entropies vector, where is included loss for each unknown author in batch
                # (for this iteration).
                softmax = self.softmax(outputs[head])
                entropies_vector = self.loss(softmax, target)
                # now, I can iterate through all unknown authors in batch for head I'm currently at
                for counter, author in enumerate(authors_order):
                    # and collect losses separately for each unknown author
                    testing_data_looses[head][author]['sum'] += entropies_vector[counter].item()
                    testing_data_looses[head][author]['counter'] += 1

        # after this, it's time to get average loss for each unknown author in each head. And ...
        # to use average loss collected earlier from training data
        max_ = -100000
        min_ = 100000
        append_to_file('output.txt', 'min max\n')
        for head in range(self.authors_size):
            for author in range(self.authors_size):
                average = testing_data_looses[head][author + 1]['sum'] / testing_data_looses[head][author + 1][
                    'counter']
                # testing_data_looses[head][author + 1]['sum'] = average - average_cross_entropies[head]['sum']
                testing_data_looses[head][author + 1]['sum'] = average
                if testing_data_looses[head][author + 1]['sum'] < min_:
                    min_ = testing_data_looses[head][author + 1]['sum']
                if testing_data_looses[head][author + 1]['sum'] > max_:
                    max_ = testing_data_looses[head][author + 1]['sum']

        diff = max_ - min_
        for head in range(self.authors_size):
            for author in range(self.authors_size):
                testing_data_looses[head][author + 1]['sum'] = (testing_data_looses[head][author + 1][
                                                                    'sum'] - min_) / diff

        results = []

        for author in range(self.authors_size):
            min_value = 1000
            min_head = -1
            for head in range(self.authors_size):
                if testing_data_looses[head][author + 1]['sum'] < min_value:
                    min_head = head
                    min_value = testing_data_looses[head][author + 1]['sum']
            results.append({'head': min_head, 'unknown_author_number': author + 1, 'loss_diff': min_value})
        append_to_file('output.txt', str(i) + '\n')
        append_to_file('output.txt', str(results))

        count = 0
        for elem in results:
            if elem['head'] + 1 == elem['unknown_author_number']:
                count += 1
        append_to_file('output.txt', '\n\ntrafieni:' + str(count))
        append_to_file('output.txt', '\n\naccuracy:' + str(count / 79))
        accuracy = count / 79
        return accuracy, average_cross_entropies[0]['sum']

    def get_average_cross_entropies(self):
        """
        Pushes known texts of each author to his head and after each sequence saves value of cross entropy.
        After whole process calculates average cross entropy for each head on known texts of its author.
        :return:
        """
        average_cross_entropies_batch_processor = OptimizedBatchProcessor(tensors_dir=self.training_tensors_path,
                                                                          batch_size=self.batch_size,
                                                                          authors_size=self.authors_size,
                                                                          timesteps=self.timesteps,
                                                                          language=self.language,
                                                                          vocab_size=self.vocab_size)
        average_cross_entropies_batch_processor.new_epoch()
        states = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

        authors_with_average_loss = self.initialize_average_training_loss_struct()

        while average_cross_entropies_batch_processor.next_batch():
            batches, target, authors_order = average_cross_entropies_batch_processor.get_results()
            batches = batches.type(torch.FloatTensor)
            outputs, _ = self.model(batches, states)
            for head in range(self.authors_size):
                softmax = self.softmax(outputs[head])
                vector = self.loss(softmax, target)
                for counter, author in enumerate(authors_order):
                    authors_with_average_loss[author - 1]['sum'] += vector[counter].item()
                    authors_with_average_loss[author - 1]['counter'] += 1

        for author in authors_with_average_loss:
            author['sum'] /= author['counter']

        return authors_with_average_loss

    def initialize_testing_loss_struct(self):
        loss_per_head_struct = []
        for head in range(self.authors_size):
            loss_per_head_struct.append({})
        for head in range(self.authors_size):
            for author in range(self.authors_size):
                loss_per_head_struct[head][author + 1] = {
                    'counter': 0,
                    'sum': 0
                }
        return loss_per_head_struct

    def initialize_average_training_loss_struct(self):
        authors_with_average_loss = []
        for author in range(self.authors_size):
            authors_with_average_loss.append({'sum': 0,
                                              'counter': 0})
        return authors_with_average_loss
