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

# Load data
with open('bestofmc.pickle', 'rb') as f:
    bestdf = pickle.load(f)

df_all = db_to_df()

df_all['post'] = df_all['post'].apply(lambda x: unidecode(x))
bestdf['post'] = bestdf['post'].apply(lambda x: unidecode(x))

# Combine 'title' and 'post' one string object named 'text', concat dfs
def _combine_text(df):
    texts = []
    for i in xrange(len(df)):
        text = df.loc[i, 'title'] + ' ' + df.loc[i, 'post']
        texts.append(text)
    return texts

texts = np.concatenate((_combine_text(df_all), _combine_text(bestdf)))

# Only keep english
engtexts = [t for t in texts if detect(t) == 'en']

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(3, 5))
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(2, 4))
X = vectorizer.fit_transform(engtexts)
feature_words = vectorizer.get_feature_names()

# Random ordering and make list of corpus sizes
rand_inds = np.random.choice(len(engtexts), len(engtexts), replace=False)
increment = 2000
corpus_sizes = range(increment, len(engtexts), increment)
corpus_sizes.append(len(engtexts))

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

def explore_latent_feat(corp=3000):
    # Explore number of topics
    mean_reconstruct_errors = []
    mses = []
    components = np.arange(5, 50)
    for n in components:
        X_ = engX[rand_inds[:3000]]
        mdl = NMF(n_components=n)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        mse = np.mean(np.square(X_ - W.dot(H)))
        mses.append(mse)
        mean_reconstruct_errors.append(mdl.reconstruction_err_)
    plt.plot(components, mean_reconstruct_errors, label='NMF reconstruction err')
    # plt.plot(components, mses, label = 'Calculated MSE')
    plt.xlabel('Number of components')
    plt.show()
