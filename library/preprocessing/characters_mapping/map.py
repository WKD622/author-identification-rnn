def map_characters(letters: dict, file) -> str:
    """
    Converts string to list of ascii characters numbers.

    Example of usage:
    path = "../../data/authors/EN001/known01.txt"
    text = ""
    f = open(path, "r")
    save_to_file("", "out.txt", map_characters(en, f))
    f.close()

    print(map_characters(en, text))
    """
    out = ""
    for line in file:
        line = line.lower()
        out += ''.join([letters[character] for character in line if character in letters])
    return out
