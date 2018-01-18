import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_weighted_average(x, weights, word_emb):
    """
    Compute the weighted average vectors
    :param word_emb: word_emb[i] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param weights: weights[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, next(iter(word_emb.values())).shape[0]))
    for i, sentence in enumerate(x):
        word_emb_mat = np.array([word_emb.get(w_idx, 0) for w_idx in sentence])
        emb[i, :] = (weights[i, :].dot(word_emb_mat) /
                     np.count_nonzero(weights[i, :]))
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(index_mat, weight_mat, word_data, params):
    """
    Compute the scores between pairs of sentences using weighted average +
    removing the projection on the first principal component
    :param index_mat: index_mat[i, :]
        are the indices of the words in the i-th sentence
    :param weight_mat: weight_mat[i, :]
        are the weights for the words in the i-th sentence
    :param word_data: word_data[i]['embedding'] is the glove word embedding for
        the word with index i
    :param params.rmpc: if >0, remove the projections of the sentence
        embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    word_emb = dict()
    for word, data in word_data.items():
        word_emb[word] = data['embedding']
    emb = get_weighted_average(index_mat, weight_mat, word_emb)
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb
