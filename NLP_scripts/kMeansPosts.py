import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode
import numpy as np
from langdetect import detect

class kMeansPosts(object):
    '''
    A class to cluster posts and output various results
    '''
    def __init__(self, df, k=5):
        '''
        INPUT:
            - df (DataFrame): DataFrame of a MissedConn object (or structured similarly)
            - k (int): number of cluster groups desired for k-means clustering
        '''
        self.df = df
        self.k = k
        self.posts = [str(unidecode(p)) for p in df['post'].values]
        self.englishposts = [p for p in self.posts if detect(p) == 'en']
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.word_vecs = None
        self.words = None
        self.km = KMeans(n_clusters=self.k)

    def fit(self):
        '''
        Constructs the doc-term matrix and corpus vocabulary
        '''
        self.word_vecs = self.vectorizer.fit_transform(self.englishposts)
        self.words = self.vectorizer.get_feature_names()
        self.km.fit(self.word_vecs)

    def print_clustered_words(self, n_words=10):
        '''
        Prints the top n_words (default=10) words that are associated with each cluster
        '''
        top_word_inds = self.km.cluster_centers_.argsort()[:, -n_words:]

        for num, ind in enumerate(top_word_inds):
            print 'Cluster {}: {}'.format(num, ', '.join(self.words[i] for i in ind))

    def print_clustered_posts(self, n_posts=3):
        '''
        Prints n_posts (default=3) random posts from each cluster
        '''
        assigned_cluster = self.km.transform(self.word_vecs).argmin(axis=1)
        for i in range(self.km.n_clusters):
            cluster = np.arange(0, self.word_vecs.shape[0])[assigned_cluster==i]
            sample_posts = np.random.choice(cluster, n_posts, replace=False)
            print '\nCluster {}:'.format(i)
            for p in sample_posts:
                print '    {}\n------------------------'.format(self.posts[p])

    def print_most_cos_sim(self, thresh=0.675):
        '''
        Prints the two posts that have the highest cosine similarity
        '''
        cos_sims = linear_kernel(self.word_vecs, self.word_vecs)

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
