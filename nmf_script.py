import pickle
import numpy as np
import pandas as pd
from manage_db import db_to_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
from langdetect import detect
from sklearn.cross_validation import train_test_split

# Load data
def load_data():
    with open('bestofmc.pickle', 'rb') as f:
        bestdf = pickle.load(f)

    df_all = db_to_df()

    bestdf['post'] = bestdf['post'].apply(lambda x: unidecode(x))
    df_all['post'] = df_all['post'].apply(lambda x: unidecode(x))

    return bestdf, df_all

# Combine 'title' and 'post' one string object named 'text', concat dfs
def _combine_text(df):
    texts = []
    for i in xrange(len(df)):
        text = df.loc[i, 'title'] + ' ' + df.loc[i, 'post']
        texts.append(text)
    return texts

def get_texts(df1, df2):
    texts = np.concatenate((_combine_text(df1), _combine_text(df2)))

    # Only keep english
    engtexts = [t for t in texts if detect(t) == 'en']
    return engtexts

# Vectorize
def vectorize(engtexts):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(3, 5))
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(2, 4))
    X = vectorizer.fit_transform(engtexts)
    feature_words = vectorizer.get_feature_names()
    return X, vectorizer, feature_words

# Random ordering and make list of corpus sizes
def randomize(X):
    rand_inds = np.random.choice(X.shape[0], X.shape[0], replace=False)
    shuffled_X = X[rand_inds]
    return shuffled_X

def _make_corpus_sizes(engtexts):
    increment = 2000
    corpus_sizes = range(increment, len(engtexts), increment)
    corpus_sizes.append(len(engtexts))
    return corpus_sizes

def cross_val(shuffled_X, n_folds=10, n_comp=5):
    size = shuffled_X.shape[0]/n_folds
    mses = []
    for fold_i in xrange(n_folds):
        X_ = shuffled_X[fold_i * size:][:size]
        # holdout = shuffled_X[:fold_i * size] + shuffled_X[(fold+i+1) * size:]

        mdl = NMF(n_components=n_comp)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        mse = np.mean(np.square(X_ - W.dot(H)))
        mses.append(mse)
    return mses

def explore_k_cross_val(shuffled_X, n_folds=10):
    all_mses = []
    mean_mses = []
    k_vals = np.arange(5, 20)
    for k in k_vals:
        mses = cross_val(shuffled_X, n_folds=n_folds, n_comp=k)
        all_mses.append(mses)
        mean_mses.append(np.mean(mses))
    plt.plot(k_vals, mean_mses)
    plt.xlabel('Number of latent features')
    plt.ylabel('Mean mse across {} folds'.format(n_folds))
    plt.show()

def explore_corpus_sizes(n_comp=5):
    # Explore corpus size
    mses = []
    for size in corpus_sizes:
        X_ = X[rand_inds[:size]]
        mdl = NMF(n_components=n_comp)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        mse = np.mean(np.square(X_ - W.dot(H)))
        mses.append(mse)
    plt.plot(corpus_sizes, mses, label = 'Calculated MSE')
    plt.xlabel('Corpus size')
    plt.show()

def explore_latent_feat(shuffled_X, corp=3000):
    # Explore number of topics
    mean_reconstruct_errors = []
    mses = []
    components = np.arange(5, 50)
    for n in components:
        X_ = shuffled_X[:3000]
        mdl = NMF(n_components=n)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        mse = np.mean(np.square(X_ - W.dot(H)))
        mses.append(mse)
        mean_reconstruct_errors.append(mdl.reconstruction_err_)
    # plt.plot(components, mean_reconstruct_errors, label='NMF reconstruction err')
    plt.plot(components, mses, label = 'Calculated MSE')
    plt.xlabel('Number of components')
    plt.ylabel('Mean reconstruction error')
    plt.show()

if __name__ == '__main__':
    bestdf, df_all = load_data()
    engtexts = get_texts(bestdf, df_all)
    X, vectorizer, feature_words = vectorize(engtexts)
    shuffled_X = randomize(X)
    # explore_k_cross_val(shuffled_X)
    explore_latent_feat(shuffled_X)
