# -*- coding: utf8 -*-
import os
import re
import string
import io
MAIN_DIR = "C:/xampp/htdocs/image-annotator/data/"


def gather_corpus():
    corpus = ""
    for set_name in os.listdir(MAIN_DIR):
        for text_file in os.listdir(MAIN_DIR + set_name + "/texts/"):
            if text_file.endswith(".txt"):
                with open(MAIN_DIR + set_name + "/texts/" + text_file, "r", encoding="utf8") as f:
                    corpus += f.read() + "\n"


def remove_punctuation(text):
    punctuation = string.punctuation + "’"
    punctuation = punctuation.replace("-", "")
    for pct in punctuation:
        text = text.replace(pct, " ")
    for pct in "„”‹›":
        text = text.replace(pct, "")
    text = text.lower()
    return text


def split_sentences(corpus):
    punctuation = "!.;?."
    sentences = re.split(punctuation, corpus)
    return sentences
def remove_empty_words(words):
    return list(filter(lambda x: x != '', words))


def split_words(sentence):
    sentence = remove_punctuation(sentence)
    words = sentence.split(" ")
    words = remove_empty_words(words)
    return words


corpus = io.open("corpus.txt", "r", encoding="utf-8").read()
sentences = split_sentences(corpus)
words = []
for sentence in sentences:
    words.extend(split_words(sentence))
with open("corpus_words.txt", 'w', encoding="utf-8") as f:
    for word in words:
        f.write(word+"\n")
