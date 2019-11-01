import sys
sys.path.append('/Users/kuba/Workspace/RNN/author-identification-rnn/')
from library.helpers.files.files_operations import (create_directory)
import matplotlib.pyplot as plt
import csv
import os

results_path = 'results'
create_directory("plots")

subfolders = [f.path for f in os.scandir("./results") if f.is_dir()]

for folder in subfolders:
    name = folder.split('/')[2]
    epochs = []
    loss = []
    accuracy = []

    with open(os.path.join(folder, "results.csv"), 'r') as csv_file:
        plots = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(plots):
            if i > 0:
                epochs.append(int(row[1]))
                loss.append(int(float(row[2])))
                accuracy.append(int(float(row[3])))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(epochs, loss, label='loss', color='g')
    ax2.plot(epochs, accuracy, label='accuracy', color='b')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    ax1.set_ylabel('loss', color='g')
    ax2.set_ylabel('accuracy', color='b')

    plt.xlabel('epochs')
    plt.title(name)

    plt.savefig(os.path.join("plots", name))
    plt.close()

