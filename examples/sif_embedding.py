import sys
sys.path.append('../src')

import pprint

import data_io
import params
import SIF_embedding

# input
# word vector file, can be downloaded from GloVe website
wordfile = '../data/glove.840B.300d.txt'
# each line is a word and its frequency
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
# the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
weightpara = 1e-3
# number of principal components to remove in SIF weighting scheme
rmpc = 1
sentences = ['this is an example sentenc'.split(),
             'this is another sentence that is slightly longer'.split()]

db = data_io.setup_db()

# weights = data_io.weights_from_file(weightfile, weightpara)
# data_io.glove_to_db(wordfile, db,  weights=weights)

# load sentences
idx_mat, weight_mat, data = data_io.prepare_data(sentences, db)

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(idx_mat, weight_mat, data, params)

pprint.pprint(embedding)
print("Cosine similarity",
      1 - scipy.spatial.distance.cosine(embedding[0, :], embedding[1, :]))
print("Euclidean distance",
      scipy.spatial.distance.euclidean(embedding[0, :], embedding[1, :]))


db.close()
