from library.preprocessing.preprocessing import Preprocessing

data_path = "../data/new/test/authors"
tensors_path = "../data/new/test/tensors"
language = 'en'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path)
