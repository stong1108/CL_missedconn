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

# 'atlanta', 'austin', 'boston', 
cities = ['chicago', 'dallas', 'denver',
    'losangeles', 'newyork', 'seattle', 'sfbay']

for city in cities:
    scrape_update(city)

# df_all = db_to_df()
# df_all.reset_index(inplace=True)
#
# def has_pic(soup):
#     if soup.findAll('figure'):
#         return 1
#     return 0
#
# import numpy as np
# raw_pages = []
# has_pics = []
# for url in df_all['url'].values:
#     try:
#         r = requests.get(url)
#         raw_pages.append(r.content)
#         soup = BeautifulSoup(r.content, 'html.parser')
#         has_pics.append(has_pic(soup))
#     except:
#         raw_pages.append(0)
#         has_pics.append(0)
#     sleep(1 + np.random.random())
