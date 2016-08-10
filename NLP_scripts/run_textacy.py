import spacy
import textacy
import pickle
from nltk.corpus import stopwords

# Load data
with open('english_missedconn.pickle', 'rb') as f:
    df = pickle.load(f)

texts = df['text'].values
utexts = [unicode(t) for t in utexts]

# Preprocessing
stop = set(stopwords.words('english'))
for doc in texts:
    newdoc = [word.lower() for word in unidecode(doc).split() if word.lower() not in stop and len(word) > 2]
    if len(newdoc) != 0:
        stopped.append(newdoc)

preprocessed = [textacy.preprocess.preprocess_text(t, fix_unicode=True,
    lowercase=True, transliterate=True, no_urls=True, no_emails=True,
    no_phone_numbers=True, no_numbers=True, no_currency_symbols=True,
    no_punct=True, no_contractions=True, no_accents=True) for t in texts]

# Make corpus
nlp = spacy.load('en')
corpus = textacy.TextCorpus.from_texts(spacy.en.English(), iter(utexts))

# Look at individual doc
doc = corpus[np.random.randint(len(utexts))]
bag = doc.as_bag_of_terms(weighting='tf', normalized=False)
for term_id, term_freq in bag.most_common(10):
    term_str = doc.spacy_stringstore[term_id]
    print('{0:>7}  {1:<13}  {2:>2}'.format(term_id, term_str, term_freq))

# represent docs as lists of words and named entities
terms_lists = (doc.as_terms_list(words=True, ngrams=(1, 3), named_entities=True)
               for doc in corpus
               if len(doc) > 200)

# convert into matrix of # documents rows by # terms columns
# weight terms by tfidf, filter out extremes
doc_term_matrix, vocab = corpus.as_doc_term_matrix(
    terms_lists, weighting='tf', normalize=True, smooth_idf=True,
    min_df=3, max_df=0.95, max_n_terms=10000)

algorithm = 'lda'
model = textacy.tm.TopicModel(algorithm, n_topics=10)
model.fit(doc_term_matrix)
model.model

for topic_idx, top_terms in model.top_topic_terms(vocab, top_n=10):
    print('topic {}:   {}'.format(topic_idx, '   '.join(top_terms)))

model.termite_plot(doc_term_matrix, vocab, topics=-1,
                   n_terms=25, sort_terms_by='seriation', rank_terms_by='topic_weight',
                   highlight_topics=None)
