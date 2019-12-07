import torch
from library.preprocessing.to_tensor.alphabets.en_alphabet import alphabet as en

path = '../data/new/test/tensors/known/EN030/EN030.pt'

tensor = torch.load(path)

for vector in tensor:
    for index, value in enumerate(vector):
        if value == 1:
            print(en[index], end='')


def decode_letter(vector):
    for index, value in enumerate(vector):
        if value == 1:
            return en[index]


def class_to_one_hot(letter):
    vector = torch.zeros(40)
    vector[letter - 1] = 1
    return vector
