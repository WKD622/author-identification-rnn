import re

KNOWN_AUTHOR = '^known(\d+)\.txt$'
UNKNOWN_AUTHOR = '^unknown\.txt$'

TEXT_NAME_CONVENTIONS = [KNOWN_AUTHOR, UNKNOWN_AUTHOR]


def check_name_convention(name: str, conventions: list):
    for convention in conventions:
        if re.match(convention, name):
            return True
    return False

#
# def _check_author_folder(path):
#     for file in os.listdir(path):
#         path = os.path.join(path, file)
#         if check_if_file(path) and check_name_convention(file, TEXT_NAME_CONVENTIONS):
#             print('right file')
#         else:
#             WrongDataStructureException(f'File {path} in author\'s directory.')
#
#
# def check_if_hidden(name: str):
#
#     return name.startswith(".")
#
#
# def check_data_structure(path: str):
#     for author in os.listdir(path):
#         path = os.path.join(path, author)
#         if check_if_directory(path):
#             print(path)
#             _check_author_folder(path)
#         elif not check_if_hidden(author):
#             # print(author)
#             # raise WrongDataStructureException('File {} in authors\' directory.'.format(path))
