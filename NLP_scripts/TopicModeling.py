import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode

class TopicModeling(object):

    def __init__(self):
        self.df = self._load_data()
        self.utexts = None
        self.texts = None
        self.stemmedtexts = None
        self.vectorizer = None
        self.X = None
        self.vocab = None
        self.shuffled_X = None
        self.groupdict = None

    def _load_data(self, picklename="english_missedconn_0812.pickle"):
        with open(picklename) as f:
            df = pickle.load(f)
        return df

    def make_texts(self):
        self.utexts = list(self.df['text'].values)
        self.texts = [unidecode(t) for t in self.utexts]
        self.stemmedtexts = [self.stemmatize_doc(t) for t in self.utexts]

    def _stemmatize(self, word):
        lmtzr = WordNetLemmatizer() # lemmatizer won't stem words ending in '-ing' unless you tell it it's a verb
        stemmer = PorterStemmer()

        if word.endswith('ing'):
            return stemmer.stem(word)
        return lmtzr.lemmatize(word)

    def stemmatize_doc(self, doc):
        return ' '.join([self._stemmatize(word) for word in doc.split()])

    def _calc_mse(self, mdl, X):
        W = mdl.fit_transform(X)
        H = mdl.components_
        mse = np.mean(np.square(X - W.dot(H)))
        return mse

    def make_groups(self, groupby_col):
        self.groupdict = dict(list(self.df.groupby(groupby_col)['text']))
        for key in self.groupdict.iterkeys():
            self.groupdict[key] = [self.stemmatize_doc(text) for text in self.groupdict[key]]

    def vectorize(self, bag_of_words=False):
        if bag_of_words:
            self.vectorizer = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))
        self.X = self.vectorizer.fit_transform(self.texts)
        self.vocab = self.vectorizer.get_feature_names()

    def randomize(self):
        rand_inds = np.random.choice(self.X.shape[0], self.X.shape[0], replace=False)
        self.shuffled_X = self.X[rand_inds]

    def make_corpus_sizes(self, increment=200):
        corpus_sizes = range(increment, len(self.texts), increment)
        corpus_sizes.append(len(self.texts))
        return corpus_sizes

    def explore_corpus_sizes(self, key, n_comp=5):
        mses = []
        self.randomize()
        corpus_sizes = self.make_corpus_sizes()
        for size in corpus_sizes:
            X_ = self.shuffled_X[:size]
            mdl = NMF(n_components=n_comp)
            mses.append(self._calc_mse(mdl, X_))
        plt.plot(corpus_sizes, mses, label=key)
        plt.xlabel('Corpus size')
        plt.ylabel('MSE')
        # plt.show()

    def cross_val(self, n_folds=7, n_comp=5):
        size = self.shuffled_X.shape[0]/n_folds
        mses = []
        for fold_i in xrange(n_folds):
            X_ = self.shuffled_X[fold_i * size:][:size]
            mdl = NMF(n_components=n_comp)
            mses.append(self._calc_mse(mdl, X_))
        return mses

    def explore_k_cross_val(self, key, n_folds=7):
        all_mses = []
        mean_mses = []
        k_vals = np.arange(5, 20)
        for k in k_vals:
            mses = self.cross_val(n_folds=n_folds, n_comp=k)
            all_mses.append(mses)
            mean_mses.append(np.mean(mses))
        plt.plot(k_vals, mean_mses, label=key)
        plt.xlabel('Number of topics')
        plt.ylabel('Mean MSE across {} folds'.format(n_folds))
        # plt.show()

    def calc_cos_sims(self):
        normalized = np.empty_like(self.X)
        for i, vec in enumerate(self.X):
            norm = np.linalg.norm(vec)
            for j, val in enumerate(vec):
                normalized[i, j] = val/norm

        sims = linear_kernel(normalized, normalized)
        return sims

    def print_nmf_words(self, n_comp=15):
        X_ = self.shuffled_X
        mdl = NMF(n_components=n_comp)
        W = mdl.fit_transform(X_)
        H = mdl.components_
        top_word_inds = np.argsort(H)[:, -10:]
        for ind, row in enumerate(top_word_inds):
            print 'Component {}: {}'.format(ind, ', '.join([feature_words[i] for i in row]))

    def explore_topics(self, key, corp=3000):
        mses = []
        components = np.arange(5, 50)
        for n in components:
            X_ = self.shuffled_X[:corp]
            mdl = NMF(n_components=n)
            mses.append(self._calc_mse(mdl, X_))
        plt.plot(components, mses, label='key')
        plt.xlabel('Number of components')
        plt.ylabel('MSE')
        # plt.show()

if __name__ == '__main__':
    tm = TopicModeling()
    tm.make_texts()
    tm.make_groups('category')
    for key in tm.groupdict.iterkeys():
        tm.texts = [tm.stemmatize_doc(t) for t in tm.groupdict[key]]
        tm.vectorize()
        print key
        tm.print_nmf_words(n_comp=15)
        print '\n--------------------------------\n'
        # tm.explore_corpus_sizes(key) # w4w too small (274)
