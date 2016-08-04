import urllib2

google_vulgar_url = 'https://gist.githubusercontent.com/ryanlewis/a37739d710ccdb4b406d/raw/0fbd315eb2900bb736609ea894b9bde8217b991a/google_twunter_lol'
resp = urllib2.urlopen(google_vulgar_url)
google_vulgar = resp.read().split('\n')

vulgar_terms_url = 'https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en'
resp = urllib2.urlopen(vulgar_terms_url)
vulgar_terms = resp.read().split('\n')



def vulgar_score(post):
    words = post.split()
    word_count = len(words)
    return sum([1. for word in words if word in google_vulgar]) / word_count
