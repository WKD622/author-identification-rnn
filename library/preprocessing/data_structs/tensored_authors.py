import os
import re

from library.preprocessing.constants import UNKNOWN, KNOWN, PATH, CONTENT, FILENAME
from library.preprocessing.files.files_operations import create_file, check_if_file, TextFileLoader, check_if_directory
from library.preprocessing.files.name_convention import check_name_convention, TEXT_NAME_CONVENTIONS, KNOWN_AUTHOR


class TensoredAuthors:
    converted_authors = {
        KNOWN: [],
        UNKNOWN: [],
    }

    def add_author(self, author: str):
        self.converted_authors.update({
            author: {
                KNOWN: None,
                UNKNOWN: None,
                PATH: ""}
        })

    def add_known(self, author: str, filename: str, content: str):
        self.converted_authors[author][KNOWN].append({
            TENSOR: content,
            FILENAME: filename
        })

    def add_unknown(self, author: str, filename: str, content: str):
        self.converted_authors[author][UNKNOWN] = {
            CONTENT: content,
            FILENAME: filename
        }

    def add_path(self, author: str, path_to_mapped: str):
        self.converted_authors[author][PATH] = path_to_mapped

    def clear(self):
        self.converted_authors = {}

    def save_to_files(self):
        pass
        # for author in self.converted_authors.keys():
        #     path = self.converted_authors[author][PATH]
        #     for known in self.converted_authors[author][KNOWN]:
        #         filename = known[FILENAME]
        #         content = known[CONTENT]
        #         create_file(filename, path, content)
        #     unknown = self.converted_authors[author][UNKNOWN]
        #     create_file(unknown[FILENAME], path, unknown[CONTENT])

    def _load_directory(self, path: str, author: str):
        self.add_author(author)
        self.add_path(author, os.path.join(path, author))

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if check_if_file(file_path) and check_name_convention(filename, TEXT_NAME_CONVENTIONS):
                text = TextFileLoader(file_path).text
                if re.match(KNOWN_AUTHOR, filename):
                    self.add_known(author, filename, content=text)
                else:
                    self.add_unknown(author, filename, content=text)

    def load_from_files(self, path):
        """
        Loads MAPPED files to memory.
        """
        self.clear()

        for author in os.listdir(path):
            directory_path = os.path.join(path, author)
            if check_if_directory(directory_path):
                self._load_directory(directory_path, author)

    def get_data(self):
        return self.converted_authors
