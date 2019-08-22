from typing import List

import torch


def char_to_tensor(char, alphabet: List):
    """
    Converts char to tensor using given alphabet.
    Alphabet has to be list of chars.
    Every sign is converted to tensor. Example:
    If alphabet is [a, b, c], tensor for a would be [1, 0, 0]
    1 is at first position because a is first in alphabet

    Another example, if alphabet is [a, b, c, d],
    for c tensor would be:
    [0, 0, 1, 0] because c is at 4th position in alphabet.

    If char doesn't exist in alphabet returned tensor contains only zeros.
    """
    tensor = torch.zeros(len(alphabet))
    for counter, element in enumerate(alphabet):
        if element == char:
            tensor[counter] = 1
    return tensor


def text_to_tensor(alphabet: List, text: str):
    """
    Converts text to tensor. Returned tensor is actually list of "smaller".
    Alphabet has to be list of chars.
    tensors, each small tensor for each char in text.
    Converting chars to tensors is caused by char_to_tensor function.
    """
    tensor = torch.LongTensor(len(text), len(alphabet))
    for counter, char in enumerate(text):
        tensor[counter] = char_to_tensor(char, alphabet)
    return tensor
