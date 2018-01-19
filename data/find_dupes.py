d = set()
GLOVE_DIM = 300
textfile = "./glove.840B.300d.txt"

with open(textfile, 'r') as f:
    for (i, line) in enumerate(f):
        if i % 10**4 == 0:
            print(i)
            # log.debug("Saving to db %d", i)
            # db.commit()
        line = line.split(' ')
        word = ' '.join(line[:-GLOVE_DIM])
        if word in d:
            print(i, word)
        d.add(word)
