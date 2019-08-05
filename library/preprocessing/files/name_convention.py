import re

KNOWN_AUTHOR = '^known(\d+)\.txt$'
UNKNOWN_AUTHOR = '^known(\d+)\.txt$'

TEXT_NAME_CONVENTIONS = [KNOWN_AUTHOR, UNKNOWN_AUTHOR]


def check_name_convention(name: str, conventions: list):
    for convention in conventions:
        if re.match(convention, name):
            return True
    return False
