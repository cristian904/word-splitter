# -*- coding: utf8 -*-
import os
import re
import string
import io
MAIN_DIR = "/Applications/XAMPP/xamppfiles/htdocs/image-annotator/data/"


def gather_corpus():
    corpus = ""
    for set_name in os.listdir(MAIN_DIR):
        if os.path.isdir(MAIN_DIR + set_name) and set_name != "Set5" and set_name != "Set7" and set_name != "Set10":
            for text_file in os.listdir(MAIN_DIR + set_name + "/texts/"):
                if text_file.endswith(".txt"):
                    with open(MAIN_DIR + set_name + "/texts/" + text_file, "r", encoding="utf8") as f:
                        corpus += f.read() + "\n"
    return corpus


def remove_punctuation(text):
    punctuation = string.punctuation + "’—῀()‹›"
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


def split_words(sentences):
    all_words = []
    for sentence in sentences:
        sentence = remove_punctuation(sentence)
        words = sentence.split(" ")
        words = remove_empty_words(words)
        all_words.extend(words)
    return all_words


corpus = gather_corpus()
words = split_words(split_sentences(remove_punctuation(corpus)))

with open("./data/corpus_words.txt", 'w', encoding="utf-8") as f:
    for word in words:
        f.write(word+"\n")
