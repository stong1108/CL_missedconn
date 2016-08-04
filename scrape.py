from MissedConn import MissedConn, HTMLTextExtractor, html_to_text
from manage_db import update_db, db_to_df
import requests
from bs4 import BeautifulSoup
from time import sleep

def scrape_update(city):
    print 'Starting ', city
    url = 'https://' + city + '.craigslist.org/search/mis'
    mc = MissedConn(url)
    mc.get_df(update=True)
    update_db(mc.df)
    print '{} done: {}'.format(city, len(mc.df))

cities = ['atlanta', 'austin', 'boston', 'chicago', 'dallas', 'denver',
'losangeles', 'miami', 'newyork', 'seattle', 'sfbay', 'washingtondc']

for city in cities:
    scrape_update(city)
