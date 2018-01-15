import sys
sys.path.append('../src')
import data_io, params, SIF_embedding

# input
wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
sentences = ['this is an example sentence', 'this is another sentence that is slightly longer']

db = data_io.setup_db()

weights = data_io.weights_from_file(weightfile, weightpara)
data_io.glove_to_db(wordfile, db,  weights=weights)

# weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# # load sentences
# x, m, _ = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
# w = data_io.seq2weight(x, m, weight4ind) # get word weights

# # set parameters
# params = params.params()
# params.rmpc = rmpc
# # get SIF embedding
# embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i

db.close()
