import os
import re
from shutil import copyfile

from library.helpers.files.files_operations import (check_if_file, TextFileLoader, append_to_file, create_directory,
                                                    create_file)
from library.helpers.files.name_convention import check_name_convention, TEXT_NAME_CONVENTIONS, KNOWN_AUTHOR

output = []
path = '../data/old/my/data'
test_path = 'test'
train_path = 'train'

create_directory(test_path)
create_directory(train_path)

for author in os.listdir(path):
    directory_path = os.path.join(path, author)
    sum_known = 0
    create_directory(os.path.join(test_path, author))
    create_directory(os.path.join(train_path, author))

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
            text_file_loader = TextFileLoader(file_path)
            text = text_file_loader.text
            if re.match(KNOWN_AUTHOR, filename):
                length = len(text_file_loader.text)
                middle = length / 2
                test_save_path = os.path.join(test_path, author)
                train_save_path = os.path.join(train_path, author)
                create_file('known01.txt', test_save_path)
                create_file('unknown.txt', test_save_path)
                create_file('known01.txt', train_save_path)
                create_file('unknown.txt', train_save_path)
                append_to_file(os.path.join(test_save_path, 'known01.txt'), text[int(middle):int(length)])
                append_to_file(os.path.join(train_save_path, 'known01.txt'), text[0:int(middle)])
                append_to_file(os.path.join(test_save_path, 'unknown.txt'), text[int(middle):int(length)])
                append_to_file(os.path.join(train_save_path, 'unknown.txt'), text[0:int(middle)])
