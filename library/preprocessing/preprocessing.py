import os
import re
import sys

from library.preprocessing.characters_mapping.map import map_characters
from library.preprocessing.conversion_to_tensor.convert import text_to_tensor
from library.preprocessing.files.files import check_if_directory, TextFileLoader, check_if_file, \
    check_if_directory_exists, create_file, remove_directory
from library.preprocessing.files.name_convention import TEXT_NAME_CONVENTIONS, check_name_convention, KNOWN_AUTHOR
import importlib
from library.preprocessing.constants import PATH, FILENAME, CONTENT, UNKNOWN, KNOWN


class Preprocessing:
    authors = []
    path = None
    alphabet = None
    mapper = None
    mapped_path = None
    tensors_path = None
    logs = None
    reduced_authors = {}

    def __init__(self, path: str, language="en", **kwargs):
        check_if_directory_exists(path)
        sys.path.insert(0, "characters_mapping/mappers")
        sys.path.insert(1, "conversion_to_tensor/alphabets")
        self.path = path
        self.mapper = importlib.import_module(language + "_mapper").charmap
        self.alphabet = importlib.import_module(language + "_alphabet").alphabet
        self.logs = kwargs.pop('logs', False)
        self.mapped_path = kwargs.pop('mapped_path', None)
        self.tensors_path = kwargs.pop('tensors_path', None)

        if self.mapped_path and check_if_directory(self.mapped_path):
            remove_directory(self.mapped_path)

    def _map_file(self):
        pass

    def map_directory(self):
        pass

    def convert_file_to_tensor(self):
        pass

    def _convert_directories_to_tensors(self):
        pass

    def logging(self, message: str):
        if self.logs:
            print(message)

    def save_mapped(self):
        for author in self.reduced_authors.keys():
            path = self.reduced_authors[author][PATH]
            for known in self.reduced_authors[author][KNOWN]:
                filename = known[FILENAME]
                content = known[CONTENT]
                create_file(filename, path, content)
            unknown = self.reduced_authors[author][UNKNOWN]
            create_file(unknown[FILENAME], path, unknown[CONTENT])

    def _map_directory(self, path, author):
        self.reduced_authors.update({
            author: {
                KNOWN: [],
                UNKNOWN: {},
                PATH: ""}
        })
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
                text_file_loader = TextFileLoader(file_path)
                mapped_text = map_characters(self.mapper, text_file_loader.text)
                self.reduced_authors[author][PATH] = os.path.join(self.mapped_path, author)
                if re.match(KNOWN_AUTHOR, filename):
                    self.reduced_authors[author][KNOWN].append({
                        CONTENT: mapped_text,
                        FILENAME: filename
                    })
                else:
                    self.reduced_authors[author][UNKNOWN] = {
                        CONTENT: mapped_text,
                        FILENAME: filename
                    }
                self.logging("Mapped " + filename)

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


path = "../../data/authors/"
mapped_path = "../../data/reduced_authors/"
p = Preprocessing(path=path, logs=False, mapped_path=mapped_path)
p.preprocess()

# check_data_structure(path)
