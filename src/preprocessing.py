from library.preprocessing.preprocessing import Preprocessing

data_path = "../data/dutch/authors/"
mapped_save_path = "../data//dutch/reduced_authors/"
mapped_source_path = "../data/dutch/reduced_authors/"
tensors_path = "../data/dutch/tensors/"
language = 'nl'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path,
                   mapped_save_path=mapped_save_path)
