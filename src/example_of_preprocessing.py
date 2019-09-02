import pprint

from library.preprocessing.preprocessing import Preprocessing

pp = pprint.PrettyPrinter(indent=3)

data_path = "../data/authors/"
mapped_save_path = "../data/reduced_authors/"
mapped_source_path = "../data/reduced_authors/"
tensors_path = "../data/tensors/"
language = 'en'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path,
                   mapped_save_path=mapped_save_path,
                   mapped_source_path=mapped_source_path,
                   batch_size=20)
pr.preprocess()
