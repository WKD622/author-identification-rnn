from library.preprocessing.preprocessing import Preprocessing

language_folder = 'dutch'
data_path = "../data/" + language_folder + "/authors/"
mapped_save_path = "../data/" + language_folder + "/reduced_authors/"
mapped_source_path = "../data/" + language_folder + "/reduced_authors/"
tensors_path = "../data/" + language_folder + "/tensors/"
language = 'nl'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path)
