import numpy as np
import statistics

GLOVE_DIM = 300
textfile = "./glove.840B.300d.txt"


def get_norms(textfile):
    with open(textfile, 'r') as f:
        for (i, line) in enumerate(f):
            if i % 10**4 == 0:
                print(i)
                # log.debug("Saving to db %d", i)
                # db.commit()
            line = line.split(' ')
            # word = ' '.join(line[:-GLOVE_DIM])
            vector = [float(x) for x in line[-GLOVE_DIM:]]
            vector = np.array(vector)
            norm = np.linalg.norm(vector)
            yield norm

median = statistics.median(get_norms(textfile))
print(median)
