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
    message = "You have to specify data source before converting to tensor"

    def __str__(self):
        return self.message
