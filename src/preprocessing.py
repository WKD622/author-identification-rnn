from library.preprocessing.preprocessing import Preprocessing

language_folder = 'english'
data_path = "../data/" + language_folder + "/authors/"
mapped_save_path = "../data/" + language_folder + "/reduced_authors/"
mapped_source_path = "../data/" + language_folder + "/reduced_authors/"
tensors_path = "../data/" + language_folder + "/tensors/"
language = 'en'
pr = Preprocessing(language=language,
                   data_path=data_path,
                   tensors_path=tensors_path)
