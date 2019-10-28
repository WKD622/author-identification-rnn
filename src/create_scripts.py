import os

import sys

sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')

from library.helpers.files.files_operations import (create_file, create_directory, check_if_directory)
from library.helpers.files.json_loader import JsonFileLoader

BEGINNING = "beginning"
NAME = "name"
NODE_NUMBERS = "node_numbers"
TASKS_PER_NODE = "tasks_per_node"
MEM_PER_CPU = "mem_per_cpu"
TIME = "time"
GRANT_NAME = "grant_name"
PARTITION = "partition"
OUTPUT = "output"
ERRORS = "errors"
HIDDEN_SIZE = "hidden_size"
NUM_LAYERS = "num_layers"
NUM_EPOCHS = "num_epochs"
BATCH_SIZE = "batch_size"
TIMESTEPS = "timesteps"
LEARNING_RATE = "learning_rate"
AUTHORS_SIZE = "authors_size"
VOCAB_SIZE = "vocab_size"
SAVE_PATH = "save_path"
TENSORS_PATH = "tensors_path"
LANGUAGE = "language"
NEW_LINE = "\n"

json_loader = JsonFileLoader('to_run.json')
json = json_loader.load()
length = len(json)


for i in range(length):
    json_el = json[i]
    results_path = 'results'
    directory_path = 'scripts'
    task_name = json_el[NAME].split()[2]
    filename = 'run-' + task_name + '.sh'
    file_path = os.path.join(directory_path, filename)
    create_file(filename=filename, path=directory_path)
    task_results_dir = os.path.join(results_path, task_name)
    if not check_if_directory(results_path):
        create_directory(results_path)
    if not check_if_directory(task_results_dir):
        create_directory(task_results_dir)
    with open(file_path, 'a') as file:
        file.write(json_el[BEGINNING])
        file.write(NEW_LINE)
        file.write(json_el[NAME])
        file.write(NEW_LINE)
        file.write(json_el[NODE_NUMBERS])
        file.write(NEW_LINE)
        file.write(json_el[TASKS_PER_NODE])
        file.write(NEW_LINE)
        file.write(json_el[MEM_PER_CPU])
        file.write(NEW_LINE)
        file.write(json_el[TIME])
        file.write(NEW_LINE)
        file.write(json_el[PARTITION])
        file.write(NEW_LINE)
        file.write(json_el[OUTPUT] + "./results/" + task_name + "/output-" + task_name + ".out")
        file.write(NEW_LINE)
        file.write(json_el[ERRORS] + "./results/" + task_name + "/errors-" + task_name + ".err")
        file.write(NEW_LINE)
        file.write('cd $SLURM_SUBMIT_DIR')
        file.write(NEW_LINE + NEW_LINE + NEW_LINE)
        file.write('module load test/pytorch/1.1.0')
        file.write(NEW_LINE)
        file.write('python3 ./rnn.py {} {} {} {} {} {} {} {} {} {} {}'.format(json_el[HIDDEN_SIZE],
                                                                              json_el[NUM_LAYERS],
                                                                              json_el[NUM_EPOCHS],
                                                                              json_el[BATCH_SIZE],
                                                                              json_el[TIMESTEPS],
                                                                              json_el[LEARNING_RATE],
                                                                              json_el[AUTHORS_SIZE],
                                                                              json_el[VOCAB_SIZE],
                                                                              os.path.join("results",
                                                                                           json_el[NAME].split()[2]),
                                                                              json_el[TENSORS_PATH],
                                                                              json_el[LANGUAGE]))
        file.close()
