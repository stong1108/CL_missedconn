import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from unidecode import unidecode
from string import punctuation
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

#--------------------------------
# Load data and preprocess
#--------------------------------
def load_data():
    with open('english_missedconn_0808.pickle', 'rb') as f:
        df = pickle.load(f)

    utexts = list(df['text'].values)
    texts = [unidecode(t) for t in utexts]

    df.reset_index(inplace=True)
    df.pop('level_0')
    cat_d = dict(list(df.groupby('category')))
    return df, utexts, texts, cat_d

def stemmatize(word):
    if word.endswith('ing'):
        return stemmer.stem(word)
    return lmtzr.lemmatize(word)

def stemmatize_doc(doc):
    return ' '.join([stemmatize(word) for word in doc.split()])

def preprocess(texts):
    # Stem & lemmatize
    lmtzr = WordNetLemmatizer() # lemmatizer won't stem words ending in '-ing' unless you tell it it's a verb
    stemmer = PorterStemmer()
    stemmedtexts = []

    stemmedtexts = [stemmatize_doc(t) for t in texts]

    nltk_stop = set(stopwords.words('english'))
    custom = set(["i'm", "it", "you", "u", "it's", "got", "get", "me", "i'd"])
    stop = nltk_stop.union(custom)

    finaltexts = []
    for doc in stemmedtexts:
        newdoc = ' '.join([word.lower() for word in doc.split() if word.lower() not in nltk_stop and len(word) > 2])
        if len(newdoc)!= 0:
            finaltexts.append(newdoc)
    return finaltexts

def mytokenize(string_of_words):
    lst = [word.strip(punctuation) for word in string_of_words.split() if len(word.strip(punctuation))>0 and word.strip(punctuation)[0].isalpha()]
    return lst

def vectorize(finaltexts):
    # Make bag of words
    vectorizer = CountVectorizer(stop_words='english', tokenizer=mytokenize, max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(finaltexts)
    feature_words = vectorizer.get_feature_names()
    return vectorizer, X, feature_words

#--------------------------------
# What do people look like?
#--------------------------------

# function for making plot
def make_plot(d, inds, name):
    df_plot = pd.DataFrame(columns=['color', 'label', 'count'])
    for key, val in d.iteritems():
        total = 0
        for cat_key in cat_d.iterkeys():
            for ind in val:
                X_col = inds[ind]
                total += np.sum(X[cat_d[cat_key].index, X_col])
            df_plot = df_plot.append({'color': key, 'label': cat_key, 'count': total}, ignore_index=True)

    # normalize
    newvals = []
    totals = df_plot.groupby('label').agg({'count': np.sum})
    for i in xrange(len(df_plot)):
        label = df_plot.loc[i, 'label']
        total = totals.ix[label]['count']
        newvals.append(total)

    df_plot['total'] = newvals
    df_plot['ratio'] = df_plot['count'] / df_plot['total']

    plt.figure(figsize=(14 ,8))
    ax = sns.barplot(x='color', y='ratio', hue='label', data=df_plot)
    ax.set(xlabel='{} description'.format(name), ylabel='Ratio')
    plt.show()

def get_inds(feature_words)
    # Collect features describing clothing
    # shorts, pants, and jeans aren't very interesting
    shirt_inds = []
    dress_inds = []
    hair_inds = []

    for i, stuff in enumerate(feature_words):
        if "shirt" == stuff.split()[-1]:
            shirt_inds.append(i)
        if "dress" in stuff.split()[-1]:
            dress_inds.append(i)
        if "hair" in stuff.split()[-1] or "brunette" in stuff or "blonde" in stuff or "redhead" in stuff:
            hair_inds.append(i)
    return shirt_inds, dress_inds, hair_inds

def plot_shirt(shirt_inds):
    # Shirts
    shirt_dict={'black': [0], 'blue': [1, 9], 'work': [2, 4], 'gray': [5, 7],
        'green': [6], 'orange': [10], 'pink': [11], 'plaid': [12], 'polo': [13],
        'purple': [14], 'red': [15], 'striped': [19], 'white': [28], 'yellow': [29]}

    make_plot(shirt_dict, shirt_inds, 'Shirt')

def plot_dress(dress_inds):
    # Dress (probably includes dress shirts?)
    dress_dict={'black': [1], 'blue': [2], 'green': [8], 'red': [9],
                'summer': [11, 12, 13], 'white': [16], 'striped': [10, 17]}

    make_plot(dress_dict, dress_inds, 'Dress')

def plot_hair(hair_inds):
    # Hair
    hair_dict={'blonde': [5, 6], 'brunette':[20, 21, 22, 32], 'red':[58, 59],
        'black':[33, 34], 'blue/green':[19, 41], 'curly':[27, 28, 73],
        'short':[62, 63, 64, 65, 66], 'long':[50, 52, 53, 54], 'facial':[37],
        'gray':[40, 42, 55, 68]}

    make_plot(hair_dict, hair_inds, 'Hair')

if __name__ == '__main__':
    df, utexts, texts, cat_d = load_data()
    finaltexts = preprocess(texts)
    vectorizer, X, feature_words = vectorize(finaltexts)
    shirt_inds, dress_inds, hair_inds = get_inds(feature_words)
    plot_shirt(shirt_inds)
    plot_dress(dress_inds)
    plot_hair(hair_inds)
