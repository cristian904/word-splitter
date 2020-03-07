# -*- coding: utf-8 -*- 
import io
import re
stopid_chars = "…«»"
map_chars = { 'ę': 'e', 'ǫ': 'o', 'ž': 'z', 'ĕ': 'e', 'ĭ': 'i', 'č': 'c', 'õ': 'o', 'š͂': 's', 'ŭ': 'u', 'ă': 'ă',  'ắ': 'ă'}
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
def read_words(path):
    with io.open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        words = list(filter(lambda x: x!='', lines))
        for i in range(len(words)):
            for key, value in map_chars.items():
                words[i] = words[i].replace(key, value)
                if not words[i].isdigit() and hasNumbers(words[i]):
                    words[i] = ''.join([i for i in words[i] if not i.isdigit()])
        return words

def get_input_characters(words):
    input_characters = set()
    for word in words:
        for char in word:
            input_characters.add(char)
    return input_characters

words = read_words("./data/corpus_words.txt")
input_characters = get_input_characters(words)
with io.open("./data/new_words.txt", "w", encoding="utf8") as f:
    for word in words:
        f.write(word+"\n")
