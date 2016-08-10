import urllib2
import pickle
from collections import defaultdict
from string import punctuation

def load_bad_words():
    with open('bad_words.pickle', 'rb') as f:
        bad_words = pickle.load(f)

    words_by_len = defaultdict(list)
    for term in bad_words:
        length = len(term.split())
        words_by_len[length].append(term)
    lengths = sorted(words_by_len.keys(), reverse=True)
    return words_by_len, lengths

def vulgar_score(post, words_by_len, lengths):
    words = [word.strip(punctuation) for word in post.split()]
    word_count = len(words)
    dirty = []

    # only check for terms that the post has enough words for
    l_ind = len(lengths)-1
    while l_ind > -1:
        if lengths[l_ind] <= word_count:
            l_ind -= 1
        else:
            lengths = lengths[(l_ind+1):]
            break

    # match bad words
    for lngth in lengths:
        for i in xrange(len(words) - lngth + 1):
            inds_to_pop = []
            if ' '.join(words[i:(i+lngth)]) in words_by_len[lngth]:
                inds_to_pop.append(i)
                dirty.append(' '.join(words[i:(i+lngth)]))
            # pop off matched words to avoid double-counting shorter bad words
            inds_to_pop.reverse()
            for ind in inds_to_pop:
                words[ind:(ind+lngth)] = []

    vulgar_count = sum([len(term.split()) for term in dirty])
    return vulgar_count, word_count, dirty

if __name__ == '__main__':

    with open('english_missedconn.pickle', 'rb') as f:
        df = pickle.load(f)

    words_by_len, lengths = load_bad_words()
    vulgar_counts = []
    word_counts = []
    dirties = []

    for text in df['text']:
        v, w, d = vulgar_score(text, words_by_len, lengths)
        vulgar_counts.append(v)
        word_counts.append(w)
        dirties.append(d)

    df['num_vulgar'] = vulgar_counts
    df['num_words'] = word_counts
    df['vulgar_terms'] = dirties

#---------------------
# Taking bad words from the internet and puttin them in a pickle!
#---------------------
# google_vulgar_url = 'https://gist.githubusercontent.com/ryanlewis/a37739d710ccdb4b406d/raw/0fbd315eb2900bb736609ea894b9bde8217b991a/google_twunter_lol'
# resp = urllib2.urlopen(google_vulgar_url)
# google_vulgar = resp.read().split('\n')
#
# # list below contains n-grams
# vulgar_terms_url = 'https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en'
# resp = urllib2.urlopen(vulgar_terms_url)
# vulgar_terms = resp.read().split('\n')
#
# # combine lists
# bad_words = list(google_vulgar)
# bad_words.extend(vulgar_terms)
# bad_words = bad_words[:-2]
#
# with open('bad_words.pickle', 'wb') as f:
#     pickle.dump(bad_words, f)
