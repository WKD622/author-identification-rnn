import os
import sys

from library.helpers.files.files_operations import (create_file)

beginning = '#!/bin/sh'
save_path = sys.argv[1]
name = sys.argv[2]
node_numbers = sys.argv[3]
tasks_per_node = sys.argv[4]
mem_per_cpu = sys.argv[5]
time = sys.argv[6]
grant_name = sys.argv[7]
partition = sys.argv[8]
output = sys.argv[9]
errors = sys.argv[10]

file_path = os.path.join(save_path, name)
create_file(save_path, name)
