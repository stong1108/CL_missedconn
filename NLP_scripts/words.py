import urllib2
import pickle
import numpy as np
from collections import defaultdict, Counter
from string import punctuation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

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

def plot_vulgar_terms(dirties, cutoff=10):
    all_dirties = []
    for lst in dirties:
        all_dirties.extend(lst)
    counts = Counter(all_dirties)
    labels = []
    vals = []
    for pair in counts.most_common():
        if pair[1] >= cutoff:
            labels.append(pair[0])
            vals.append(pair[1])
    indexes = np.arange(len(labels))
    width = 1

    plt.figure(figsize=(16, 12))
    plt.bar(indexes, vals, width)
    plt.xticks(indexes+width*0.5, labels, rotation=90)
    plt.show()

def make_scatter(df, col='category'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, (name, group) in enumerate(df.groupby(col)):
        cscale = group.ratio / 0.20
        axes[i].scatter(group.index, group.ratio, c=-cscale, cmap='inferno', alpha=0.3)
        axes[i].axhline(np.mean(group.ratio),0, len(df), c='k')
        axes[i].set_title(name)
        axes[i].set_xlim(0, len(df))
        axes[i].set_yscale('log')
    plt.show()

def make_box(df, col='category'):
    fig = plt.figure(figsize=(16, 12))
    df_cut = df[df['ratio'] > 0.015][['category', 'ratio']]
    sns.violinplot(x=col, y='ratio', data=df_cut)
    plt.show()

def make_violin_age(df, col='category'):
    fig = plt.figure(figsize=(16, 12))
    sns.violinplot(x=col, y='age', data=df)
    plt.show()

def make_age_hist(df, col='category'):
    age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    df['age'].hist(normed=True, by=df[col], bins=age_bins)
    plt.show()

def print_vulgarity_box(df):
    df['ratio'] = df['num_vulgar'] / df['num_words']
    age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    df['age_grp'] = pd.cut(df['age'], bins=age_bins, include_lowest=True)
    sns.violinplot(x='age_grp', y='ratio', data=df)
    plt.show()

def print_most_vulgar(df, by='num_vulgar'):
    df['ratio'] = df['num_vulgar'] / df['num_words']
    print df.loc[np.argmax(df[by]), 'text']

def print_vulgarity_means(df, old_age=90):
    df['ratio'] = df['num_vulgar'] / df['num_words']
    df_no_cat = df.loc[df['category'].isnull()]
    df_no_age = df.loc[(df['age'] > 1) & (df['age'] < old_age)]
    df_no_both = df.loc[(df['age'] > 1) & (df['age'] < old_age) & df['category'].isnull()]

    with_cats = df.groupby('category').agg({'ratio': np.mean})
    d = dict((ind, with_cats.loc[ind, 'ratio']) for ind in with_cats.index.values)
    d['no_category'] = df_no_cat['ratio'].mean()
    d['no_age'] = df_no_age['ratio'].mean()
    d['no_age_or_cat'] = df_no_both['ratio'].mean()

    v_sorted = sorted(d.values())
    k_sorted = []

    for v in v_sorted:
        for k in d.iterkeys():
            if d[k] == v:
                k_sorted.append(k)

    for i in xrange(len(k_sorted)):
        print '\n{:<15}: {:0.4f}'.format(k_sorted[i], v_sorted[i])


def clean_cities(df):
    df['city'] = df['city'].apply(lambda x: str(x))
    for i, city in enumerate(df['city']):
        if 'sfbay' in city:
            df.loc[i, 'city'] = 'sfbay'
        elif 'washington' in city:
            df.loc[i, 'city'] = 'washingtondc'
        elif 'missedconnnections' in city:
            end_ind = (len('missedconnections') + city.index('missedconnections'))
            df.loc[i, 'city'] = city[end_ind:]
    return df

def make_states(df):
    state_dict = \
    {'arizona': ['tucson', 'phoenix'],
    'canada': ['winnipeg', 'toronto', 'vancouver', 'ottawa', 'edmonton', 'calgary', 'montreal'],
    'colorado': ['denver', 'fortcollins', 'boulder', 'colosprings'],
    'connecticut': ['hartford', 'newhaven'],
    'england': ['london'],
    'florida': ['miami', 'orlando', 'sarasota', 'tampabay', 'pensacola'],
    'georgia': ['atlanta'],
    'hawaii': ['hawaii'],
    'illinois': ['chicago'],
    'indiana': ['indianapolis'],
    'iowa': ['iowacity'],
    'kansas': ['wichita', 'kansascity'],
    'kentucky': ['lexington'],
    'louisiana': ['neworleans'],
    'maine': ['maine'],
    'maryland': ['baltimore', 'washingtondc', 'washington'],
    'massachussetts': ['boston'],
    'michigan': ['annarbor', 'grandrapids', 'lansing', 'detroitmetro'],
    'minnesota': ['minneapolis'],
    'nebraska': ['lincoln', 'omaha'],
    'newmexico': ['albuquerque'],
    'newyork': ['newyork', 'buffalo', 'albany', 'rochester', 'northjersey'],
    'norcal': ['sfbay', 'sacramento', 'humboldt', 'modesto', 'eugene'],
    'nc': ['fayetteville', 'asheville', 'raleigh'],
    'ohio': ['columbus', 'cleveland'],
    'oklahoma': ['oklahomacity'],
    'oregon': ['portland', 'salem', 'medford'],
    'pennsylvania': ['philadelphia', 'pittsburgh', 'allentown'],
    'rhodeisland': ['rhodeisland'],
    'socal': ['sandiego', 'losangeles', 'orangeco', 'orangecounty', 'santabarbara'],
    'sc': ['charleston'],
    'texas': ['dallas', 'austin', 'houston', 'sanantonio'],
    'tennessee': ['nashville', 'clarksville'],
    'utah': ['saltlake'],
    'virginia': ['blacksburg', 'norfolk', 'richmond'],
    'washdc': ['washingtondc', 'washington'],
    'washington': ['seattle', 'bellingham', 'spokane'],
    'wisconsin': ['eauclaire', 'greenbay', 'madison', 'milwaukee']}

    city_dict = {}

    for state, cities in state_dict.iteritems():
        for city in cities:
            city_dict[city] = state

    df['state'] = df['city'].apply(lambda x: city_dict[x])
    return df

if __name__ == '__main__':

    with open('english_missedconn_0808.pickle', 'rb') as f:
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

    make_plot(dirties)

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
