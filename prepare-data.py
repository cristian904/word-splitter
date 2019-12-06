# -*- coding: utf-8 -*- 
import io
stopid_chars = "…«»"
map_chars = { 'ę': 'e', 'ǫ': 'o', 'ž': 'z', 'ĕ': 'e', 'ĭ': 'i', 'č': 'c', 'õ': 'o', 'š͂': 's', 'ŭ': 'u'}
def read_words(path):
    with io.open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        words = list(filter(lambda x: x!='', lines))
        for i in range(len(words)):
            # for char in stopid_chars:
            #     words[i] = words[i].replace(char, "")
            for key, value in map_chars.items():
                words[i] = words[i].replace(key.decode("utf8"), value)
        return words

def get_input_characters(words):
    input_characters = set()
    for word in words:
        for char in word:
            input_characters.add(char)
    return input_characters

words = read_words("corpus_words.txt")
input_characters = get_input_characters(words)
with io.open("new_words.txt", "w", encoding="utf8") as f:
    for word in words:
        f.write(word+"\n")
