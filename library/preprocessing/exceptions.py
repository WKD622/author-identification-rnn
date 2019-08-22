class WrongDataStructureException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class NotMappedDataException(Exception):
    message = "You have to map data before getting it"

    def __str__(self):
        return self.message


class NoDataSourceSpecified(Exception):
    message = "You have to specify data source | EXAMPLE: (data_path=/some/path)"

    def __str__(self):
        return self.message


class NoLanguageSpecified(Exception):
    message = "You have to specify language | EXAMPLE: (language=\"en\")"

    def __str__(self):
        return self.message
