class FileLoader:
    file = ""
    file_lines = []

    def __init__(self, path):
        self.path = path

    def load_file(self):
        with open(self.path) as file:
            for line in file:
                self.file += line
                self.file_lines.append(line.rstrip())
        return self.file, self.file_lines


def save_to_file(path: str, name: str, text: str):
    text_file = open(path + name, "w+")
    text_file.write(text)
    text_file.close()
