import os


class FileLoader:
    file = ""
    file_lines = []

    def __init__(self, path):
        self.path = path
        self._load_file()

    def _load_file(self):
        with open(self.path) as file:
            for line in file:
                self.file += line
                self.file_lines.append(line.rstrip())
        return self.file, self.file_lines


def save_to_file(path: str, filename: str, text: str):
    text_file = open(os.path.join(path, filename))
    text_file.write(text)
    text_file.close()


def check_if_directory(path: str) -> bool:
    """
    Returns True if path points at directory, otherwise False.
    """
    return os.path.isdir(path)


def check_if_file(path: str) -> bool:
    """
    Returns True if path points at file, otherwise False.
    """
    return os.path.isfile(path)


def check_if_directory_exists(path: str):
    """
    Rises FileNotFoundError if directory under given path does not exist.
    :param path: e
    :return:
    """
    if not os.path.isdir(path):
        raise FileNotFoundError()
