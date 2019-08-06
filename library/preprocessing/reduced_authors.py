import os

from library.preprocessing.constants import KNOWN, UNKNOWN, PATH, FILENAME, CONTENT
from library.preprocessing.files.files_operations import create_file


class ReducedAuthors:
    reduced_authors = {}

    def add_author(self, author: str):
        self.reduced_authors.update({
            author: {
                KNOWN: [],
                UNKNOWN: {},
                PATH: ""}
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
        self.reduced_authors[author][PATH] = os.path.join(path_to_mapped, author)

    def clear(self):
        self.reduced_authors = {}

    def save_to_files(self):
        for author in self.reduced_authors.keys():
            path = self.reduced_authors[author][PATH]
            for known in self.reduced_authors[author][KNOWN]:
                filename = known[FILENAME]
                content = known[CONTENT]
                create_file(filename, path, content)
            unknown = self.reduced_authors[author][UNKNOWN]
            create_file(unknown[FILENAME], path, unknown[CONTENT])

    def get_data(self):
        return self.reduced_authors