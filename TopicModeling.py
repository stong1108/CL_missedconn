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


class TopicModeling(object):

    def __init__(self):
        self.bestdf, self.df_all = self._load_data()
        self.texts = self._get_texts()
        self.vectorizer = None
        self.X = None
        self.vocab = None
        self.shuffled_X = None

    def _load_data(self):
        with open('bestofmc.pickle', 'rb') as f:
            bestdf = pickle.load(f)

        df_all = db_to_df()

        bestdf['post'] = bestdf['post'].apply(lambda x: unidecode(x))
        df_all['post'] = df_all['post'].apply(lambda x: unidecode(x))

        return bestdf, df_all

    def _combine_text(self, df):
        texts = []
        for i in xrange(len(df)):
            text = df.loc[i, 'title'] + ' ' + df.loc[i, 'post']
            texts.append(text)
        return texts

    def _get_texts(self):
        texts = np.concatenate((self._combine_text(self.bestdf), self._combine_text(self.df_all)))

        # Only keep english
        engtexts = [t for t in texts if detect(t) == 'en']
        return engtexts

    def vectorize(self, ngrams=0):
        if ngrams!=0:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=ngrams)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.X = self.vectorizer.fit_transform(self.texts)
        self.vocab = self.vectorizer.get_feature_names()

    def randomize(self):
        rand_inds = np.random.choice(self.X.shape[0], self.X.shape[0], replace=False)
        self.shuffled_X = self.X[rand_inds]

    def make_corpus_sizes(self):
        increment = 2000
        corpus_sizes = range(increment, len(self.texts), increment)
        corpus_sizes.append(len(self.texts))
        return corpus_sizes

    def cross_val(self, n_folds=10, n_comp=5):
        size = self.shuffled_X.shape[0]/n_folds
        mses = []
        for fold_i in xrange(n_folds):
            X_ = self.shuffled_X[fold_i * size:][:size]
            # holdout = shuffled_X[:fold_i * size] + shuffled_X[(fold+i+1) * size:]

            mdl = NMF(n_components=n_comp)
            W = mdl.fit_transform(X_)
            H = mdl.components_
            mse = np.mean(np.square(X_ - W.dot(H)))
            mses.append(mse)
        return mses

    def print_words(self, corp=3000, n_comp=10, l1_ratio=1, alpha=0.1):
        X_ = self.shuffled_X[:corp]
        mdl = NMF(n_components=n_comp, l1_ratio=l1_ratio, alpha=alpha)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        top_word_inds = np.argsort(H)[:, -10:]
        for ind, row in enumerate(top_word_inds):
            print 'Component {}: {}'.format(ind, ', '.join([feature_words[i] for i in row]))

    def explore_k_cross_val(self, n_folds=10):
        all_mses = []
        mean_mses = []
        k_vals = np.arange(5, 20)
        for k in k_vals:
            mses = cross_val(self.shuffled_X, n_folds=n_folds, n_comp=k)
            all_mses.append(mses)
            mean_mses.append(np.mean(mses))
        plt.plot(k_vals, mean_mses)
        plt.xlabel('Number of latent features')
        plt.ylabel('Mean mse across {} folds'.format(n_folds))
        plt.show()

    def explore_corpus_sizes(self, n_comp=5):
        # Explore corpus size
        mses = []
        for size in self.make_corpus_sizes():
            X_ = self.X[rand_inds[:size]]
            mdl = NMF(n_components=n_comp)
            W = mdl.fit_transform(X_)
            H = mdl.components_
            mse = np.mean(np.square(X_ - W.dot(H)))
            mses.append(mse)
        plt.plot(corpus_sizes, mses, label = 'Calculated MSE')
        plt.xlabel('Corpus size')
        plt.show()

    def explore_latent_feat(self, corp=3000):
        # Explore number of topics
        mean_reconstruct_errors = []
        mses = []
        components = np.arange(5, 50)
        for n in components:
            X_ = self.shuffled_X[:corp]
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
    tm = TopicModeling()
    tm.vectorize()
    tm.randomize()
    tm.print_words()
