import json as _


class JsonFileLoader:
    json = None

    def __init__(self, path):
        self.path = path

    def load(self):
        with open(self.path) as json_file:
            self.json = _.load(json_file)
        return self.json
