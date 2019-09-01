import importlib
import os
import re
import sys

import torch

from library.preprocessing.chars_mapping.map import map_characters
from library.preprocessing.constants import (KNOWN, UNKNOWN, LANGUAGE, DATA_PATH, MAPPED_SOURCE_PATH, MAPPED_SAVE_PATH,
                                             REDUCED_AUTHORS, TENSORS_PATH, TENSORS)
from library.preprocessing.data_structs.reduced_authors import ReducedAuthors
from library.preprocessing.exceptions import NotMappedDataException, NoDataSourceSpecified, NoLanguageSpecified
from library.preprocessing.files.files_operations import (check_if_directory, TextFileLoader, check_if_file,
                                                          check_if_directory_exists, remove_directory)
from library.preprocessing.files.name_convention import TEXT_NAME_CONVENTIONS, check_name_convention, KNOWN_AUTHOR
from library.preprocessing.to_tensor.convert import text_to_tensor

from library.preprocessing.batch_processing.batching import BatchProcessor


class CharactersMapper:
    """
    """
    data_path = None
    mapper = None
    mapped_save_path = None
    mapped = False
    reduced_authors = ReducedAuthors()

    def __init__(self, language: str, **kwargs):
        self.data_path = kwargs.get(DATA_PATH)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/chars_mapping/mappers")
        self.mapper = importlib.import_module(language + "_mapper").charmap
        self.mapped_save_path = kwargs.get(MAPPED_SAVE_PATH)
        if self.data_path:
            self.map()

    def save_mapped(self):
        self.reduced_authors.save_to_files()

    def _map_directory(self, path: str, author: str):
        """
        Maps directory from given path.
        """
        self.reduced_authors.add_author(author)
        if self.mapped_save_path:
            self.reduced_authors.add_path(author, os.path.join(self.mapped_save_path, author))
        else:
            self.reduced_authors.add_path(author, os.path.join(REDUCED_AUTHORS, author))

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
        if self.mapped_save_path and check_if_directory(self.mapped_save_path):
            remove_directory(self.mapped_save_path)
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
        if self.mapped_save_path:
            self.save_mapped()

    def get_data(self):
        if self.mapped:
            return self.reduced_authors.get_data()
        else:
            raise NotMappedDataException()


class ToTensor:
    alphabet = None
    tensors_path = None
    mapped_source_path = None
    reduced_authors = None
    converted = False

    def __init__(self, language: str, **kwargs):
        self.mapped_source_path = kwargs.get(MAPPED_SOURCE_PATH)
        if self.mapped_source_path:
            check_if_directory_exists(self.mapped_source_path)
        sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/to_tensor/alphabets")
        self.alphabet = importlib.import_module(language + "_alphabet").alphabet
        self.tensors_path = kwargs.get(TENSORS_PATH)
        if not self.tensors_path:
            self.tensors_path = TENSORS
        self.reduced_authors = kwargs.get(REDUCED_AUTHORS)
        self.convert()

    def preparation(self):
        """
        Preparation before mapping. If there is mapped_path specified removes old content from it,
        and then clears data structure to be able to be filled with new data.
        """
        if check_if_directory(self.tensors_path) and self.tensors_path:
            remove_directory(self.tensors_path)

    def save_tensor_to_file(self, tensor, path: str, filename: str):
        os.makedirs(path)
        full_path = os.path.join(path, filename + '.pt')
        torch.save(tensor, full_path)

    def _convert_to_tensors(self, reduced_authors: ReducedAuthors):
        batch_processor = BatchProcessor()
        for author in reduced_authors.get_data().keys():
            known_tensor = text_to_tensor(self.alphabet, reduced_authors.get_author_merged_known(author))
            unknown_tensor = text_to_tensor(self.alphabet, reduced_authors.get_author_unknown(author))
            batch_processor.set_params(known_tensor)
            known_batches = batch_processor.get_batches()
            batch_processor.set_params(unknown_tensor)
            unknown_batches = batch_processor.get_batches()
            known_path = os.path.join(self.tensors_path, KNOWN, author)
            unknown_path = os.path.join(self.tensors_path, UNKNOWN, author)
            self.save_tensor_to_file(tensor=known_tensor, path=known_path, filename=author)
            self.save_tensor_to_file(tensor=unknown_tensor, path=unknown_path, filename=author)


    def convert(self):
        reduced_authors = ReducedAuthors()
        if self.mapped_source_path:
            print("using hard drive for creating tensors")
            reduced_authors.load_from_files(self.mapped_source_path)
        elif self.reduced_authors:
            print("using ram for creating tensors")
            reduced_authors.load_dict(self.reduced_authors)
        else:
            raise NoDataSourceSpecified()
        self.preparation()
        self._convert_to_tensors(reduced_authors)


class Preprocessing:
    kwargs = None
    language = None
    data_path = None

    def check_kwargs(self, kwargs):
        if not kwargs.get(LANGUAGE):
            raise NoLanguageSpecified()

    def __init__(self, **kwargs):
        self.check_kwargs(kwargs)
        self.language = kwargs.get(LANGUAGE, 'en')
        self.data_path = kwargs.get(DATA_PATH)
        self.kwargs = kwargs
        self.preprocess()

    def preprocess(self):
        characters_mapper = CharactersMapper(**self.kwargs)
        if self.data_path:
            self.kwargs.update({REDUCED_AUTHORS: characters_mapper.get_data()})
        ToTensor(**self.kwargs)