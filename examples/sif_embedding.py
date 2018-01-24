import sys
sys.path.append('../src')

import pprint
import scipy
import scipy.spatial
import scipy.spatial.distance

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
sentences = ['this is an example sentence'.split(),
             'this is another sentence that is slightly longer'.split(),
             'and now for someting completely different'.split(),
             'gorillas are found in Africa'.split()]

sentences = [
    'the lion is the king of the jungle'.split(),
    'tigers hunt alone at night'.split(),
    'long live the emperor'.split(),
    'we call him little bobby tables'.split()]

# import pickle
# sentences = pickle.load(open('tiger.pd', 'rb'))

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

with open('svdump.pd', 'wb') as f:
    pickle.dump(embedding, f)
pprint.pprint(list(enumerate(sentences)))

print("Cosine dist"),
pprint.pprint(scipy.spatial.distance.squareform(
              scipy.spatial.distance.pdist(embedding, 'cosine')))
print("Euclidean dist"),
pprint.pprint(scipy.spatial.distance.squareform(
              scipy.spatial.distance.pdist(embedding, 'euclidean')))


db.close()
