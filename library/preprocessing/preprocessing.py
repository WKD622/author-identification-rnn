import os

from library.preprocessing.files.files import check_if_directory, FileLoader, check_if_file, check_if_directory_exists
from library.preprocessing.files.name_convention import TEXT_NAME_CONVENTIONS, check_name_convention


class Preprocessing:
    authors = []
    path = None

    def __init__(self, path: str):
        check_if_directory_exists(path)
        self.path = path

    def save_to_files(self, path: str):
        check_if_directory_exists(path)

    def _map_file(self):
        pass

    def map_directory(self):
        pass

    def convert_file_to_tensor(self):
        pass

    def _convert_directories_to_tensors(self):
        pass

    def _preprocess_text(self, text):
        print(text)

    def _preprocess_directory(self, path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if check_if_file(file_path) and check_name_convention(file, TEXT_NAME_CONVENTIONS):
                file_loader = FileLoader(file_path)
                self._preprocess_text(file_loader.file)

    def preprocess(self):
        for author in os.listdir(self.path):
            directory_path = os.path.join(self.path, author)
            if check_if_directory(directory_path):
                self._preprocess_directory(directory_path)


path = "../../data/authors/"
p = Preprocessing(path=path)
p.preprocess()
