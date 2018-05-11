#!/usr/bin/env python
from src import data_io
import sys

DB_FILE = sys.argv[1]
db = data_io.setup_db(DB_FILE)
weights = data_io.weights_from_file("./auxiliary_data/enwiki_vocab_min200.txt")
data_io.glove_to_db(sys.argv[2], db, weights)
