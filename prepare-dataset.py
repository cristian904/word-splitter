import io

with io.open("new_words.txt", "r", encoding="utf8") as f:
    words = f.readlines()

current_pos = 0
i=3
with io.open("dataset.txt", "w", encoding="utf8") as f:
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
        
        f.write(concatenated_words + " " + target + '\n')

    
    
