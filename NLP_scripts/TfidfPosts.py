import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode
from langdetect import detect

class TfidfPosts(object):
    '''
    A class for outputting various results from TF-IDF doc-term construction
    '''
    def __init__(self, df):
        '''
        INPUT:
            - df (DataFrame): DataFrame of a MissedConn object (or structured similarly)
        '''
        self.df = df
        self.posts = [unidecode(df['title'].values[i]) + ' ' + \
            unidecode(df['post'].values[i]) for i in xrange(len(df))]
        self.englishposts = [p for p in self.posts if detect(p) == 'en']
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.word_vecs = None
        self.words = None

    def fit(self):
        '''
        Constructs the doc-term matrix and corpus vocabulary
        '''
        self.word_vecs = self.vectorizer.fit_transform(self.englishposts)
        self.words = self.vectorizer.get_feature_names()

    def find_posts(self, word):
        '''
        Returns an iterable of posts given a word to search for
        '''
        word_ind = self.words.index(word)
        word_col = self.word_vecs.getcol(word_ind).toarray().flatten()
        match_inds = np.nonzero(word_col)[0]
        ordered_match_inds = match_inds[np.argsort(word_col[match_inds])[::-1]]
        return iter(np.array(self.englishposts)[ordered_match_inds])

    def print_most_cos_sim(self, thresh=0.675):
        '''
        Prints the two posts that have the highest cosine similarity
        '''
        # normalize self.word_vecs
        normalized = np.empty_like(self.word_vecs)
        for i, vec in enumerate(self.word_vecs):
            for j, val in enumerate(vec):
                normalized[i, j] = val / np.linalg.norm(vec)

        cos_sims = linear_kernel(normalized, normalized)

        # Initialize max_sim = 0, only consider cos sims under threshold
        # so we know we're not recording a post compared with itself (1.0)
        max_cos_sim = 0.0
        thr = thresh

        # Find max_cos_sim
        for i, j in enumerate(cos_sims):
            for k, l in enumerate(j):
                if (float(l) >= max_cos_sim) and (float(l) < thr):
                    max_cos_sim = float(l)

        # Find indices of max_cos_sim
        double_break = False
        for i, j in enumerate(cos_sims):
            for k, l in enumerate(j):
                if float(l) == max_cos_sim:
                    ind1, ind2 = i, k
                    double_break = True
                    break
            if double_break:
                break

        print 'Posts with highest cosine similarity ({:.3f}):\n\nPost {}:\n{}\
            \n\nPost {}:\n{}'.format(max_cos_sim, ind1, self.posts[ind1],
                                    ind2, self.posts[ind2])
