def map_characters(mapper: dict, text: str) -> str:
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
    for line in text:
        line = line.lower()
        out += ''.join([mapper[character] for character in line if character in mapper])
    return out
