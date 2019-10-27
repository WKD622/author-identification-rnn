import os
import shutil


class TextFileLoader:
    """
    Used for easy file usage. During initialization you have to pass path to file
    in constructor. Example:

    file_path = "/example/path"
    text_file_loader = TextFileLoader(file_path)

    and you have this text in str form:
    text_file_loader.text

    and in list form, where elements are file lines (also str):
    text_file_loader.text_lines
    """
    text = ""
    text_lines = []

    def __init__(self, path):
        self.path = path
        self._load_file()

    def _load_file(self):
        with open(self.path) as file:
            for line in file:
                self.text += line
                self.text_lines.append(line.rstrip())
        return self.text, self.text_lines


def save_to_file(path: str, filename: str, content: str):
    """
    Saves file under given path and with given filename and content.
    """
    text_file = open(os.path.join(path, filename))
    text_file.write(content)
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


def create_file(filename: str, path: str, content=""):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename), 'w') as new_file:
        new_file.write(content)


def remove_directory(path):
    shutil.rmtree(path)


def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def append_to_file(path, to_append):
    with open(path, 'a') as file:
        file.write(to_append)
        file.close()
