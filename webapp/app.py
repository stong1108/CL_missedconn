import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import Form
from wtforms import validators, TextField
from flask_paginate import Pagination

app = Flask(__name__)
app.config.from_object('config')

# Forms
class SearchForm(Form):
    word = TextField('search', validators=[validators.DataRequired()])

#-----------------------
# Load data, search fcn
with open('english_missedconn_0812.pickle', 'rb') as f:
    df = pickle.load(f)

vectorizer = TfidfVectorizer(stop_words='english')

word_vecs = vectorizer.fit_transform(df['text'].values)
vocab = vectorizer.get_feature_names()

def find_posts(word):
    '''
    Returns an iterable of posts given a word to search for
    '''
    word_ind = vocab.index(str(word))
    word_col = word_vecs.getcol(word_ind).toarray().flatten()
    match_inds = np.nonzero(word_col)[0]
    ordered_match_inds = match_inds[np.argsort(word_col[match_inds])[::-1]]
    return ordered_match_inds

#-----------------------

# App pages
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Hello!')

@app.route('/about')
def about():
    return render_template('about.html', title="About")

@app.route('/error', methods=['GET', 'POST'])
@app.route('/error/<word>', methods=['GET', 'POST'])
def error(word):
    return render_template('error.html', word=word)

@app.route('/maps', methods=['GET', 'POST'])
def maps():
    return render_template('map_cities.html', title='Maps')

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        return redirect(url_for('result', word=request.form['word'].lower(), page=1))
    return render_template('search.html', title='Search', form=form)

@app.route('/result', methods=['GET', 'POST'])
@app.route('/result/<word>/<int:page>', methods=['GET', 'POST'])
def result(word, page):
    # get matches
    try:
        ordered_match_inds = find_posts(word)
        num = ordered_match_inds.shape[0]
    except ValueError:
        return redirect(url_for('error', word=word))

    pagination = Pagination(page=page, per_page=1, total=num, record_name='post')
    missed_conn = df.iloc[ordered_match_inds]
    items = [dict(missed_conn.iloc[i]) for i in xrange(num)]
    ind = page - 1
    item = items[ind]

    return render_template('result.html', keys=word, page=page, word=word,
        item=item, num=num, pagination=pagination, title='Results')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
