import os


class Preprocessing:
    authors = []
    path = None

    @staticmethod
    def _check_if_directory_exists(path: str):
        """
        Rises FileNotFoundError if directory under given path does not exist.
        :param path: e
        :return:
        """
        if not os.path.isdir(path):
            raise FileNotFoundError()

    def __init__(self, path: str):
        self._check_if_directory_exists(path)
        self.path = path

    def save_to_files(self, path: str):
        self._check_if_directory_exists(path)

    def _map_file(self):
        pass

    def map_directory(self):
        pass

    def convert_file_to_tensor(self):
        pass

    def _convert_directories_to_tensors(self):
        pass

    def _preprocess_file(self):
        pass

    def _preprocess_directory(self):
        pass

    def preprocess(self):
        for author in os.listdir(self.path):
            if
            for text in os.listdir(self.path + author):
                print(text)
        else:
            pass


path = "../../data/authors/"
p = Preprocessing(path=path)
p.preprocess()
