import os
import re
import sys

from library.preprocessing.chars_mapping.map import map_characters
from library.preprocessing.constants import KNOWN, UNKNOWN
from library.preprocessing.files.files_operations import check_if_directory, TextFileLoader, check_if_file, \
    check_if_directory_exists, remove_directory
from library.preprocessing.files.name_convention import TEXT_NAME_CONVENTIONS, check_name_convention, KNOWN_AUTHOR
import importlib
from library.preprocessing.reduced_authors import ReducedAuthors


class Preprocessing:
    authors = []
    path = None
    alphabet = None
    mapper = None
    mapped_path = None
    tensors_path = None
    logs = None
    reduced_authors = ReducedAuthors()

    def __init__(self, path: str, language="en", **kwargs):
        check_if_directory_exists(path)
        sys.path.insert(0, "chars_mapping/mappers")
        sys.path.insert(1, "to_tensor/alphabets")
        self.path = path
        self.mapper = importlib.import_module(language + "_mapper").charmap
        self.alphabet = importlib.import_module(language + "_alphabet").alphabet
        self.logs = kwargs.pop('logs', False)
        self.mapped_path = kwargs.pop('mapped_path', None)
        self.tensors_path = kwargs.pop('tensors_path', None)

        if self.mapped_path and check_if_directory(self.mapped_path):
            remove_directory(self.mapped_path)

    def _convert_to_tensors(self):
        for_conversion = self.reduced_authors.get_data()
        for author in for_conversion.keys():
            for known in for_conversion[author][KNOWN]:
                pass
            unknown = self.reduced_authors[UNKNOWN]

    def logging(self, message: str):
        if self.logs:
            print(message)

    def save_mapped(self):
        self.reduced_authors.save_to_files()

    def _map_directory(self, path: str, author: str):
        """
        Maps directory from given path.
        """
        self.reduced_authors.add_author(author)
        if self.mapped_path:
            self.reduced_authors.add_path(author, os.path.join(self.mapped_path, author))

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
                text_file_loader = TextFileLoader(file_path)
                mapped_text = map_characters(self.mapper, text_file_loader.text)
                if re.match(KNOWN_AUTHOR, filename):
                    self.reduced_authors.add_known(author, filename, content=mapped_text)
                else:
                    self.reduced_authors.add_unknown(author, filename, content=mapped_text)

    def _map_directories(self):
        """
        Iterates through all directories and runs _map_directory method on each.
        """
        for author in os.listdir(self.path):
            directory_path = os.path.join(self.path, author)
            if check_if_directory(directory_path):
                self._map_directory(directory_path, author)
        if self.mapped_path:
            self.save_mapped()

    def preprocess(self):
        self._map_directories()
        self._convert_to_tensors()


path = "../../data/authors/"
mapped_path = "../../data/reduced_authors/"
p = Preprocessing(path=path, logs=False, mapped_path=mapped_path)
p.preprocess()

# check_data_structure(path)
