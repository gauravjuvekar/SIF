import os
import numpy as np
import pickle
from tree import tree
# from theano import config

import sqlite3
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

GLOVE_DIM = 300
DB_FILE = "../data/sif.db"


def encode(s):
    return s
    # return s.encode('ascii', errors='backslashreplace')


def setup_db(f=DB_FILE):
    db = sqlite3.connect(DB_FILE)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS word_indexes(
            word TEXT
                NOT NULL
                PRIMARY KEY,
            idx INTEGER
                UNIQUE
                NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sif_embeddings(
            idx INTEGER
                NOT NULL
                PRIMARY KEY
                REFERENCES word_indexes(idx),
            embedding BLOB,
            weight REAL
        );
        """)
    db.execute("PRAGMA synchronous = OFF")
    db.execute("PRAGMA journal_mode = MEMORY")
    return db


def embedding_to_bytes(vector):
    return np.array(vector).tobytes()


def embedding_from_bytes(bytestring):
    return np.frombuffer(bytestring, dtype=float)


def get_max_glove_word_len(textfile):
    max_word_len = 0
    max_word_len_i = 0
    the_word = None
    with open(textfile, 'r') as f:
        for (i, line) in enumerate(f):
            line = line.decode().rstrip().split(' ')
            word = ' '.join(line[:-GLOVE_DIM])
            if len(word) > max_word_len:
                the_word = word
                max_word_len = len(word)
                max_word_len_i = i
    return max_word_len, max_word_len_i, the_word


def glove_to_db(textfile, db, weights=None):
    if weights is None:
        weights = dict()
    with open(textfile, 'r') as f:
        for (i, line) in enumerate(f):
            if i % 10**4 == 0:
                log.debug("Saving to db %d", i)
                db.commit()
            line = line.split(' ')
            word = ' '.join(line[:-GLOVE_DIM])
            try:
                word = encode(word)
            except TypeError as e:
                raise Exception("Error encoding word") from e

            vector = [float(x) for x in line[-GLOVE_DIM:]]
            vector = np.array(vector)
            try:
                db.execute(
                    "INSERT OR ABORT INTO "
                    "word_indexes(word, idx) VALUES (?, ?);",
                    (word, i))
            except sqlite3.IntegrityError as e:
                log.critical(
                    "IntegrityError: Possible duplicate entry in Glove"
                    "embeddings for word %r, line %d" % (word, i))
            else:
                weight = weights.get(word, 1.0)
                db.execute(
                    """INSERT INTO sif_embeddings(idx, embedding, weight)
                       VALUES (?, ?, ?)""",
                    (i, embedding_to_bytes(vector), weight))

    db.commit()


def weights_from_file(weightfile, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    count_sum = 0
    word_weight_dict = {}
    with open(weightfile) as f:
        for i, line in enumerate(f):
            if i % 100 == 0:
                log.debug("Reading weight %d", i)
            line = line.strip()
            if(len(line) > 0):
                line = line.split()
                if(len(line) == 2):
                    word = encode(line[0])
                    count = float(line[1])
                    word_weight_dict[word] = count
                    count_sum += count

    for word, weight in word_weight_dict.items():
        word_weight_dict[word] = a / (a + (weight / count_sum))
    return word_weight_dict


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def lookup_indexes(words, db):
    words = [encode(word.lower()) for word in words]
    words = [word.replace("#", "")
             if len(word) and word.startswith('#') else word
             for word in words]
    d = dict(
        db.execute(
            "SELECT word, idx FROM word_indexes "
            "WHERE word in (" + ', '.join(['?'] * len(words)) + ");",
            words))
    return [d.get(word, None) for word in words]


def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1

def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    for i in p2:
        X2.append(lookupIDX(words, i))
    return X1, X2

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return list(zip(list(range(len(minibatches))), minibatches))

def getSimEntDataset(f,words,task):
    examples = []
    with open(f, 'r') as data:
        for i in data:
            i = i.strip()
            if(len(i) > 0):
                i = i.split('\t')
                if len(i) == 3:
                    if task == "sim":
                        e = (tree(i[0], words), tree(i[1], words), float(i[2]))
                        examples.append(e)
                    elif task == "ent":
                        e = (tree(i[0], words), tree(i[1], words), i[2])
                        examples.append(e)
                    else:
                        raise ValueError('Params.traintype not set correctly.')

                else:
                    print(i)
    return examples

def getSentimentDataset(f,words):
    examples = []
    with open(f, 'r') as data:
        for i in data:
            i = i.strip()
            if(len(i) > 0):
                i = i.split('\t')
                if len(i) == 2:
                    e = (tree(i[0], words), i[1])
                    examples.append(e)
                else:
                    print(i)
    return examples

def getDataSim(batch, nout):
    g1 = []
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    if nout <= 0:
        return (scores, g1x, g1mask, g2x, g2mask)

    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype='float32')
    return (scores, g1x, g1mask, g2x, g2mask)

def getDataEntailment(batch):
    g1, g2 = [], []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(3)
        label = i[2].strip()
        if label == "CONTRADICTION":
            temp[0] = 1
        if label == "NEUTRAL":
            temp[1] = 1
        if label == "ENTAILMENT":
            temp[2] = 1
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype='float32')
    return (scores, g1x, g1mask, g2x, g2mask)

def getDataSentiment(batch):
    g1 = []
    for i in batch:
        g1.append(i[0].embeddings)

    g1x, g1mask = prepare_data(g1)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[1].strip()
        if label == "0":
            temp[0] = 1
        if label == "1":
            temp[1] = 1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores, dtype='float32')
    return (scores, g1x, g1mask)

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def sentiment2idx(sentiment_file, words):
    """
    Read sentiment data file, output array of word indices that can be fed into the algorithms.
    :param sentiment_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, golds. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location), golds[i] is the label (0 or 1) for sentence i.
    """
    with open(sentiment_file, 'r') as f:
        golds = []
        seq1 = []
        for i in f:
            i = i.split("\t")
            p1, score = i[0], int(i[1])  # score are labels 0 and 1
            X1 = getSeq(p1, words)
            seq1.append(X1)
            golds.append(score)
        x1, m1 = prepare_data(seq1)
        return x1, m1, golds

def sim2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the score for pair i (float). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    with open(sim_file,'r') as f:
        golds = []
        seq1 = []
        seq2 = []
        for i in f:
            i = i.split("\t")
            p1, p2, score = i[0], i[1], float(i[2])
            X1, X2 = getSeqs(p1, p2, words)
            seq1.append(X1)
            seq2.append(X2)
            golds.append(score)
        x1, m1 = prepare_data(seq1)
        x2, m2 = prepare_data(seq2)
        return x1, m1, x2, m2, golds

def entailment2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the label for pair i (CONTRADICTION NEUTRAL ENTAILMENT). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    with open(sim_file, 'r') as f:
        golds = []
        seq1 = []
        seq2 = []
        for i in f:
            i = i.split("\t")
            p1, p2, score = i[0], i[1], i[2]
            X1, X2 = getSeqs(p1, p2, words)
            seq1.append(X1)
            seq2.append(X2)
            golds.append(score)
        x1, m1 = prepare_data(seq1)
        x2, m2 = prepare_data(seq2)
        return x1, m1, x2, m2, golds

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight
