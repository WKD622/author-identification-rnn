class DirectoryNotSpecified(Exception):
    message = "You have to specify directory to do the processing."

    def __str__(self):
        return self.message
