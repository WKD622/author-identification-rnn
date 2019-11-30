from library.preprocessing.preprocessing import Preprocessing

data_path = "../data/old/english/authors"
tensors_path = "../data/old/english/tensors"
language = 'en'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path)
