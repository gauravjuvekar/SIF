import itertools
import numpy as np
import statistics
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

        CREATE TABLE IF NOT EXISTS meta(
            key TEXT
                NOT NULL
                PRIMARY KEY,
            value_text TEXT,
            value_int INTEGER,
            value_float REAL
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
    norms = []
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
                norms.append(np.linalg.norm(vector))
                db.execute(
                    """INSERT INTO sif_embeddings(idx, embedding, weight)
                       VALUES (?, ?, ?)""",
                    (i, embedding_to_bytes(vector), weight))
    median = statistics.median(norms)
    db.execute("INSERT INTO meta(key, value_float) VALUES (?, ?)",
               ("glove_median_norm", median))
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


def sentence_to_indices(sentence, db):
    words = [encode(word.lower()) for word in sentence]
    # Because hashtags arent words or something
    words = [word.replace("#", "")
             if len(word) and word.startswith('#') else word
             for word in words]


def get_indices_for_tokens(words, db):
    db.commit()
    db.execute("CREATE TEMPORARY TABLE temporary_tokens( "
               "    word TEXT PRIMARY KEY NOT NULL"
               ");")
    db.executemany("INSERT OR IGNORE INTO temporary_tokens(word) VALUES (?);",
                   [(word,) for word in words])

    d = dict(
        db.execute(
            "SELECT word, idx FROM word_indexes WHERE word IN "
            "(SELECT word FROM temporary_tokens);"))
    db.execute("DROP TABLE temporary_tokens;")
    db.rollback()
    return d


def get_data_for_indices(indices, db, d=None):
    indices = list(indices)
    indices_set = set(indices) - set((None,))
    if d is None:
        d = dict()
    query = db.execute(
        "SELECT idx, weight, embedding FROM sif_embeddings "
        "WHERE idx IN (" + ', '.join(['?'] * len(indices_set)) + ");",
        list(indices_set))
    for idx, weight, embedding_bytes in query:
        d[idx] = {'weight': weight,
                  'embedding': embedding_from_bytes(embedding_bytes)}

    glove_norm = None
    for idx in indices:
        if idx not in d:
            if glove_norm is None:
                glove_norm = db.execute("SELECT value_float FROM meta "
                                        "WHERE key =='glove_median_norm'")
                glove_norm = glove_norm.fetchone()[0]
            randvec = np.random.uniform(low=-1, high=1, size=GLOVE_DIM)
            randvec = glove_norm * (randvec / np.linalg.norm(randvec))
            d[idx] = {'weight': 1.0, 'embedding': randvec}
    return d


def prepare_data(list_of_token_lists, db):
    lengths = [len(s) for s in list_of_token_lists]
    n_samples = len(list_of_token_lists)

    flatten = [x for y in list_of_token_lists for x in y]
    indices = get_indices_for_tokens(flatten, db)

    # replace tokens with indices or unique negative indices if not found
    neg_count = itertools.count(-1, -1)
    for token in flatten:
        if token not in indices:
            indices[token] = next(neg_count)

    list_of_indices_lists = [[indices[word] for word in sentence]
                             for sentence in list_of_token_lists]
    flatten = [x for y in list_of_indices_lists for x in y]
    data = get_data_for_indices(flatten, db)

    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_weight = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, sentence in enumerate(list_of_indices_lists):
        x[idx, :lengths[idx]] = sentence
        x_weight[idx, :lengths[idx]] = [data[word]['weight']
                                        for word in sentence]
    return x, x_weight, data
