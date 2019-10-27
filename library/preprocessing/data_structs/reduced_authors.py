import os
import re
from typing import List

from library.preprocessing.constants import KNOWN, UNKNOWN, PATH, FILENAME, CONTENT
from library.helpers.files.files_operations import create_file, check_if_directory, check_if_file, TextFileLoader
from library.helpers.files.name_convention import check_name_convention, TEXT_NAME_CONVENTIONS, KNOWN_AUTHOR


class ReducedAuthors:
    """
    Class which stores reduced texts as objects, it has all helpful methods to do that.
    """
    reduced_authors = {}

    def add_author(self, author: str):
        self.reduced_authors.update({
            author: {
                KNOWN: [],
                UNKNOWN: {},
                PATH: ""
            }
        })

    def add_known(self, author: str, filename: str, content: str):
        self.reduced_authors[author][KNOWN].append({
            CONTENT: content,
            FILENAME: filename
        })

    def add_unknown(self, author: str, filename: str, content: str):
        self.reduced_authors[author][UNKNOWN] = {
            CONTENT: content,
            FILENAME: filename
        }

    def add_path(self, author: str, path_to_mapped: str):
        self.reduced_authors[author][PATH] = path_to_mapped

    def clear(self):
        self.reduced_authors = {}

    def save_to_files(self):
        for author in self.reduced_authors.keys():
            path = self.get_author_path(author)
            for known in self.get_author_known(author):
                filename = known[FILENAME]
                content = known[CONTENT]
                create_file(filename, path, content)
            unknown = self.get_author_unknown(author)
            create_file(unknown[FILENAME], path, unknown[CONTENT])

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

    def load_dict(self, reduced_authors):
        self.reduced_authors = reduced_authors

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
        return self.reduced_authors

    def get_author(self, author):
        return self.reduced_authors[author]

    def get_author_known(self, author: str) -> List:
        return self.reduced_authors[author][KNOWN]

    def get_author_unknown(self, author: str) -> str:
        return self.reduced_authors[author][UNKNOWN]

    def get_author_path(self, author: str) -> str:
        return self.reduced_authors[author][PATH]

    def get_author_merged_known(self, author: str) -> str:
        known = self.get_author_known(author)
        merged = ""
        for text in known:
            merged += text[CONTENT]
        return merged
