import io
from random import uniform

with io.open("./data/new_words.txt", "r", encoding="utf8") as f:
    words = f.readlines()

current_pos = 0
i=3
train = io.open("./data/train_set.txt", "w", encoding="utf8")
test = io.open("./data/test_set.txt", "w", encoding="utf8")
val = io.open("./data/validation_set.txt", "w", encoding="utf8")
while current_pos < len(words) - 1:
    current_words = words[current_pos:current_pos+i]
    current_words = list(map(lambda x: x.strip(), current_words))
    concatenated_words = ''.join(current_words)
    current_pos += i
    i+=1
    if i == 8:
        i = 3

    target = ""
    for word in current_words:
        if len(word)>1:
            target += 'i'
            target += ''.join(['m'] * (len(word) - 2))
            target += 's'
        else:
            target += 'c'
    prob = uniform(0, 1)
    if prob < 0.85: 
        train.write('|'.join(current_words) + " " +concatenated_words + " " + target + '\n')
    elif prob >= 0.85 and prob < 0.93:
        test.write('|'.join(current_words) + " " +concatenated_words + " " + target + '\n')
    else:
        val.write('|'.join(current_words) + " " +concatenated_words + " " + target + '\n')
train.close()
test.close()
val.close()
    
    
