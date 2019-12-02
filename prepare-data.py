import os
MAIN_DIR = "C:/xampp/htdocs/image-annotator/data/"


def gather_corpus():
    corpus = ""
    for set_name in os.listdir(MAIN_DIR):
        for text_file in os.listdir(MAIN_DIR + set_name + "/texts/"):
            if text_file.endswith(".txt"):
                with open(MAIN_DIR + set_name + "/texts/" + text_file, "r", encoding="utf8") as f:
                    corpus += f.read() + "\n"


def remove_punctuation(corpus):
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    punctuation = punctuation.replace("-", "")
    for pct in punctuation:
        corpus = corpus.replace(pct, " ")
    for pct in "„”‹›":
        corpus = corpus.replace(pct, "")
    corpus = corpus.lower()
    with open("corpus_no_pcts.txt", "w", encoding="utf8") as f:
        f.write(corpus)

def 

# remove_punctuation(open("corpus.txt", "r", encoding="utf-8").read())
