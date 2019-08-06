from typing import List

import torch


def char_to_tensor(char, alphabet):
    tensor = torch.zeros(len(alphabet))
    for counter, element in enumerate(alphabet):
        if element == char:
            tensor[counter] = 1
    return tensor


def text_to_tensor(alphabet: List, text: str):
    tensor = torch.LongTensor(len(text), len(alphabet))
    for counter, char in enumerate(text):
        tensor[counter] = char_to_tensor(char, alphabet)
    return tensor
#
# from library.preprocessing.to_tensor.alphabets.en import alphabet as en_alphabet
# from library.files.files import save_to_file, FileLoader
# from library.preprocessing.chars_mapping.map import map_characters
# from library.preprocessing.chars_mapping.mappers.en import charmap as en
# path = "../../../data/authors/EN001/known01.txt"
# text = ""
# f = open(path, "r")
# save_to_file("", "out.txt", map_characters(en, f))
# f.close()
#
# fl = FileLoader("out.txt")
# file, _ = fl.load_file()
# torch.set_printoptions(profile="full")
# print(text_to_tensor(file, en_alphabet))
