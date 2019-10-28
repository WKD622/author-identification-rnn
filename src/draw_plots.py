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
                print(row)
                epochs.append(int(row[1]))
                loss.append(int(float(row[2])))
                accuracy.append(int(float(row[3])))

    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, accuracy, label='accuracy')

    plt.xlabel('epochs')
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join("plots", name))
