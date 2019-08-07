import os
import re
import sys

import torch

from library.preprocessing.chars_mapping.map import map_characters
from library.preprocessing.constants import KNOWN, UNKNOWN, PATH, FILENAME, CONTENT
from library.preprocessing.data_structs.tensored_authors import TensoredAuthors
from library.preprocessing.exceptions import NotMappedDataException, NoDataSourceSpecified
from library.preprocessing.files.files_operations import check_if_directory, TextFileLoader, check_if_file, \
    check_if_directory_exists, remove_directory, create_file
from library.preprocessing.files.name_convention import TEXT_NAME_CONVENTIONS, check_name_convention, KNOWN_AUTHOR
import importlib
from library.preprocessing.data_structs.reduced_authors import ReducedAuthors
from library.preprocessing.to_tensor.convert import text_to_tensor


class CharactersMapper:
    data_path = None
    mapper = None
    mapped_path = None
    logs = None
    reduced_authors = ReducedAuthors()
    mapped = False

    def __init__(self, data_path: str, language="en", **kwargs):
        check_if_directory_exists(data_path)
        sys.path.insert(0, "chars_mapping/mappers")
        self.data_path = data_path
        self.mapper = importlib.import_module(language + "_mapper").charmap
        self.logs = kwargs.get('logs', False)
        self.mapped_path = kwargs.get('mapped_path')
        self.map()

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
        else:
            self.reduced_authors.add_path(author, os.path.join('reduced_authors', author))

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
                text_file_loader = TextFileLoader(file_path)
                mapped_text = map_characters(self.mapper, text_file_loader.text)
                if re.match(KNOWN_AUTHOR, filename):
                    self.reduced_authors.add_known(author, filename, content=mapped_text)
                else:
                    self.reduced_authors.add_unknown(author, filename, content=mapped_text)

    def preparation(self):
        """
        Preparation before mapping. If there is mapped_path specified removes old content from it,
        and then clears data structure to be able to be filled with new data.
        """
        if self.mapped_path and check_if_directory(self.mapped_path):
            remove_directory(self.mapped_path)
        self.reduced_authors.clear()

    def map(self):
        """
        Iterates through all directories and runs _map_directory method on each.
        """
        self.preparation()
        for author in os.listdir(self.data_path):
            directory_path = os.path.join(self.data_path, author)
            if check_if_directory(directory_path):
                self._map_directory(directory_path, author)
        self.mapped = True
        if self.mapped_path:
            self.save_mapped()

    def get_data(self):
        if self.mapped:
            return self.reduced_authors.get_data()
        else:
            raise NotMappedDataException()


class ToTensor:
    alphabet = None
    tensors_path = None
    mapped_path = None
    reduced_authors = None
    converted = False

    def __init__(self, language="en", **kwargs):
        self.mapped_path = kwargs.get('mapped_path')
        if self.mapped_path:
            check_if_directory_exists(mapped_path)
        sys.path.insert(1, "to_tensor/alphabets")
        self.alphabet = importlib.import_module(language + "_alphabet").alphabet
        self.logs = kwargs.get('logs', False)
        self.tensors_path = kwargs.get('tensors_path')
        if not self.tensors_path:
            self.tensors_path = 'tensors'
        self.reduced_authors = kwargs.get('reduced_authors')
        self.convert()

    def save_tensor_to_file(self, tensor, path: str, filename: str):
        os.makedirs(path)
        full_path = os.path.join(path, filename + '.pt')
        torch.save(tensor, full_path)

    def _convert_to_tensors(self, reduced_authors: ReducedAuthors):
        for author in reduced_authors.get_data().keys():
            known_tensor = text_to_tensor(self.alphabet, reduced_authors.get_author_merged_known(author))
            unknown_tensor = text_to_tensor(self.alphabet, reduced_authors.get_author_unknown(author))
            known_path = os.path.join(self.tensors_path, KNOWN, author)
            unknown_path = os.path.join(self.tensors_path, UNKNOWN, author)
            self.save_tensor_to_file(tensor=known_tensor, path=known_path, filename=author)
            self.save_tensor_to_file(tensor=unknown_tensor, path=unknown_path, filename=author)

    def convert(self):
        reduced_authors = ReducedAuthors()
        if self.reduced_authors:
            reduced_authors.load_dict(self.reduced_authors)
        elif self.mapped_path:
            reduced_authors.load_from_files(self.mapped_path)
        else:
            raise NoDataSourceSpecified()
        self._convert_to_tensors(reduced_authors)


class Preprocessing:
    def preprocess(self):
        pass

import pprint
pp = pprint.PrettyPrinter(indent=3)

data_path = "../../data/authors/"
mapped_path = "../../data/reduced_authors/"
tensors_path = "../../data/tensors/"
language = "en"
ch = CharactersMapper(data_path=data_path, language=language, mapped_path=mapped_path)
conv = ToTensor(language=language, reduced_authors=ch.get_data(), tensors_path=tensors_path)
