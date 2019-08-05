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
            raise FileNotFoundError

    def __init__(self, path: str):
        os.path.isdir(path)
        self.path = path

    def save_to_files(self, path: str):
        self._check_if_directory_exists(path)

    def _preprocess_file(self):
        pass

    def _preprocess_directory(self):
        pass

    def preprocess(self):
        if self.path:
            for author in os.listdir(self.path):
                if filename.endswith(".asm") or filename.endswith(".py"):
                    # print(os.path.join(directory, filename))
                    continue
                else:
                    continue
        else:
