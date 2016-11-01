# Craigslist Missed Connections - NLP

**Write-up of the Content Analysis project lives here:**
http://http://taketwodatablog.com/craigslists-missed-connections/

### Contents

**TopicModeling.py**<br>
Main script for topic modeling using NMF

**vulgar_words.py**<br>
Script for exploring vulgarity of posts by age or category.
Vulgarity is determined by matching of terms in these compiled dirty (NSFW) words:

* [List 1: https://gist.github.com/ryanlewis/a37739d710ccdb4b406d](https://gist.github.com/ryanlewis/a37739d710ccdb4b406d)<br>
* [List 2: https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en)

**word2vec_svd.py**<br>
Script for topic modeling using Google or Scrapy word vectors (LSA)

**find_trends.py**<br>
Script to explore commonly mentioned:
+ shirt colors
+ dress colors
+ hair types

**TfidfPosts.py**<br>
Class file for an object to explore results from TF-IDF doc-term construction

**kMeansPosts.py**<br>
Class file for an object to explore results from K-means clustering
