import urllib2
import pickle
from collections import defaultdict

# Load data
# with open('english_missedconn_0808.pickle', 'rb') as f:
#     df = pickle.load(f)

with open('bad_words.pickle', 'rb') as f:
    bad_words = pickle.load(f)

words_by_len = defaultdict(list)
for term in bad_words:
    length = len(term.split())
    words_by_len[length].append(term)
lengths = sorted(words_by_len.keys(), reverse=True)

def vulgar_score(post):
    words = post.split()
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
