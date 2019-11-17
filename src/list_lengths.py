import os
import re

from library.helpers.files.files_operations import check_if_file, TextFileLoader, append_to_file
from library.helpers.files.name_convention import check_name_convention, TEXT_NAME_CONVENTIONS, KNOWN_AUTHOR

output = []
path = '../data/my/data'
output_file_path = 'en_train.txt'
for author in os.listdir(path):
    directory_path = os.path.join(path, author)
    sum_known = 0
    first_chars = ''
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
            text_file_loader = TextFileLoader(file_path)
            if re.match(KNOWN_AUTHOR, filename):
                first_chars = first_chars + text_file_loader.text[:100]
                sum_known += len(text_file_loader.text)
    output.append((author, sum_known, first_chars.replace("\n", "").replace(" ", "").replace("\t", "")))

output = sorted(output, key=lambda tup: tup[1], reverse=True)

for tup in output:
    append_to_file(output_file_path,
                   str(tup[0]) + "  " + str(tup[1]) + "     " + str(tup[2]) + "\n")
