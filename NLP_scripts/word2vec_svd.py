import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from unidecode import unidecode
from string import punctuation
import spacy.en
from gensim.models.word2vec import Word2Vec, Vocab
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

def load_data():
    with open('english_missedconn_0808.pickle', 'rb') as f:
        df = pickle.load(f)
    groupdict = dict(list(df.groupby('category')['text']))
    return df, groupdict

def make_model(type='gensim'):
    if type=='google':
        model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    else:
        nlu = spacy.en.English()
        model = Word2Vec(size=300)

        for i, lex in enumerate(nlu.vocab):
            model.vocab[lex.orth_] = Vocab(index=i, count=None)
            model.index2word.append(lex.orth_)

        model.syn0norm = np.asarray(map(lambda x: x.repvec, nlu.vocab))
        model.syn0 = np.asarray(map(lambda x: x.repvec, nlu.vocab))
    return model

def stemmatize(word):
    lmtzr = WordNetLemmatizer()
    stemmer = PorterStemmer()
    if word.endswith('ing'):
        return stemmer.stem(word)
    return lmtzr.lemmatize(word)

def stemmatize_doc(doc):
    return ' '.join([stemmatize(word) for word in doc.split()])

def mytokenize(string_of_words):
    nltk_stops = set(stopwords.words('english'))
    custom = set(["i'm", "it", "you", "it's", "got", "get", "me", 'want', 'like',
        'time', 'saw', 'craigslist', 'look', 'know', 'wa', 'number', 'think',
        'knew', 'wanted', 'today', 'say', 'said', 'craigs', 'list', 'year',
        'went', 'think', 'd', 's', 'nt', 'phone', 'number', 'talk', 'talked',
        'walked', 'seat', 'right', 'missed', 'connection', 'guy', 'girl',
        'woman', 'man', 'women', 'men', 'flag', 'number', "i'm", "i've", "i'll",
        "it's", "don't", "won't", "didn't", "wasn't", "time", "said", "flag",
        "n", "mc", "got", "didn't", "just", "did", "wa", "make", "no", "isn't",
        "today", "day", "ha", "haha", "lol", "yes", "see", "saw", "b", "c", "d",
        "make", "missed", "connection", "connections", "cl", "say", "someth",
        "craigslist", "craig", "did", "do", "ad", "need", "just", "look",
        "like", "you're", "can't", "he's", "want", "know", "knew", "tell",
        "told", "think", "thing", "thought", "really", "maybe", "im", "she's",
        "craig's", "amp", "want", "blah", "went", "talk", "talked", "meet",
        "met", "let", "lets", "let's", "hey", "hi", "i'd", "sit", "l", "sat",
        "you've", "your", "you're", "you'll", "youre", "seen", "couldn't", "the"
        "can't", "re", "re:re", "doesn't", "try", "tried", "don't", "won't",
        "we'll", "they're", "theyre"])

    stops = nltk_stops.union(custom)
    lst = [word.strip(punctuation) for word in string_of_words.split() if \
        len(word.strip(punctuation))>0 and word.strip(punctuation)[0].isalpha() \
        and not any([p in word.strip(punctuation) for p in punctuation])\
        and word.strip(punctuation) not in stops]
    return lst

def get_doc_vec(doc, model):
    tokenized = mytokenize(doc)
    temp = np.zeros((len(tokenized), 300))
    for i, token in enumerate(tokenized):
        try:
            temp[i, :] = model[token]
        except KeyError:
            temp[i, :] = np.zeros((1, 300))
    summed = np.sum(temp, axis=0)
    result = np.empty_like(summed)
    for j, val in enumerate(summed):
        result[j] = val / np.linalg.norm(summed)
    return result

def print_topic(mytexts, model, n_topics=15):
    X = make_matrix(mytexts)
    mdl = TruncatedSVD(n_components=n_topics)
    U = mdl.fit_transform(X)
    V = mdl.components_

    normalized_V = np.empty_like(V)
    for i, component in enumerate(V):
        for j, val in enumerate(component):
            normalized_V[i, j] = val / np.linalg.norm(component)

    for i, component in enumerate(normalized_V):
        print '\nComponent {}: '.format(i)
        for match in model.most_similar([component], topn=5):
            print '\n\t{}'.format(match)

def make_matrix(mytexts):
    X = np.zeros((len(mytexts), 300))
    for i, doc in enumerate(mytexts):
        X[i] = get_doc_vec(doc, model)
    return X

def compare_doc_word(doc, model):
    tokenized = mytokenize(doc)
    temp = np.zeros((len(tokenized), 300))
    for i, token in enumerate(tokenized):
        try:
            temp[i, :] = model[token]
        except KeyError:
            temp[i, :] = np.zeros((1, 300))
    summed = np.sum(temp, axis=0)
    result = np.empty_like(summed)
    for j, val in enumerate(summed):
        result[j] = val / np.linalg.norm(summed)
    sims = [linear_kernel(result, word) for word in temp]
    return max(sims)

def explore_num_topics(mytexts):
    X = make_matrix(mytexts)
    topics = np.arange(10, 35)
    errors = []
    for t in topics:
        mdl = TruncatedSVD(n_components=t)
        U = mdl.fit_transform(X)
        V = mdl.components_
        D = np.diag(mdl.explained_variance_)
        mse = np.mean(np.square(X - np.dot(U, np.dot(D, V))))
        errors.append(mse)
    plt.figure(figsize=(12, 8))
    plt.plot(topics, errors)
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean squared error')
    plt.show()

if __name__ == '__main__':
    df, groupdict = load_data()
    model = make_model("google")
    mytexts = groupdict['m4m'].values
    print_topic(mytexts, model)
