from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
import numpy as np
from unidecode import unidecode
from datetime import datetime
from manage_db import most_recent_post_dt
from HTMLParser import HTMLParser

class MissedConn(object):
    '''
    A class to scrape Craigslist Missed Connections for a given city
    '''
    def __init__(self, starturl):
        '''
        INPUT:
            - starturl (str): url of a city's Missed Connections page
            (e.g. "https://sfbay.craigslist.org/search/mis")
        '''
        self.starturl = starturl
        self.city = self._get_city()
        self.record_dt = None
        self.df = pd.DataFrame(columns=['title', 'category', 'post_dt', 'latitude',
                    'longitude', 'neighborhood', 'extra', 'age', 'post', 'url',
                    'record_dt', 'city', 'raw_page', 'has_pic'])
        self._baseurl = self.starturl[:self.starturl.find('/search')]
        self._hrefs = []
        self._nhoods = None

    def get_df(self, update=False):
        '''
        Populates and returns self.df with Missed Connection post info

        INPUT:
            - update (bool): If True, only retrieve "new" posts for the
            MissedConn object's city. If False, retrieves all posts.

            "new" is determined by the city's last post_dt in the missedconn
            table of the cl_missedconn database.
        '''
        # Go to missed connections!
        self.record_dt = datetime.now()
        soup = self._look_at_page(self.starturl)
        if update:
            self._hrefs = []
            self.df = pd.DataFrame(columns=['title', 'category', 'post_dt', 'latitude',
                        'longitude', 'neighborhood', 'extra', 'age', 'post', 'url',
                        'record_dt', 'city', 'raw_page', 'has_pic'])

        # Fix neighborhood stuff later...
        # -------------
        #Get neighborhoods
        # nh_content = soup.findAll('input', {'class': 'multi_checkbox multi_nh'})
        # self._nhoods = set(str(nh.text.strip()) for nh in nh_content)
        # -------------

        # Get post links (and 'next' link if it exists)
        self._hrefs.extend(self._get_hrefs(soup))
        nextlink = soup.find('a', {'class': 'button next'})

        while nextlink:
            oldlink = nextlink
            nexturl = self._baseurl + str(nextlink['href'])
            nextsoup = self._look_at_page(nexturl)
            nextlink = nextsoup.find('a', {'class': 'button next'})
            if oldlink == nextlink:
                break
            self._hrefs.extend(self._get_hrefs(nextsoup))

        # Get rid of duplicate hrefs and maintain ordering
        lst = set()
        self._hrefs = [x for x in self._hrefs if str(x) not in lst and not lst.add(str(x))]

        # Populate df with post info
        for href in self._hrefs:
            url = self._baseurl + href
            soup = self._look_at_page(url)
            if soup.find('div', {'class': 'removed'}): # Skip "removed" posts
                continue
            if update:
                # Only get posts that were posted after most recent record collection
                cutoff_dt = most_recent_post_dt(self.city)
                try:
                    if cutoff_dt > self._get_datetime(soup):
                        break
                except (ValueError, IndexError):
                    print url
                    continue
            try:
                self._get_info(url)
            except (ValueError, IndexError):
                print url
                continue
            sleep(5 + 3*np.random.random()) # to emulate a real person

    def _get_city(self):
        if self.starturl.startswith('http'):
            ind1 = 2 + self.starturl.find('//')
        else:
            ind1 = 0
        ind2 = self.starturl.find('.craigslist')
        return self.starturl[ind1:ind2]

    def _get_info(self, url):
        soup = self._look_at_page(url)
        raw_page = requests.get(url).content
        title, category = self._get_title_and_cat(soup)
        dt = self._get_datetime(soup)
        latitude, longitude = self._get_coords(soup)
        neighborhood = self._get_nhood(soup)
        extra, age = self._get_extra_age(soup)
        post = self._get_post(soup)
        has_pic = self._has_pic(soup)

        # Collect info into dict to append to df
        d = {'title': title, 'category': category, 'post_dt': dt, 'latitude': latitude,
            'longitude': longitude, 'neighborhood': neighborhood, 'extra': extra,
            'age': age, 'post': post, 'url': url, 'record_dt': self.record_dt,
            'city': self.city, 'raw_page': raw_page, 'has_pic': has_pic}

        self.df = self.df.append(d, ignore_index=True)

    def _get_title_and_cat(self, soup):
        temp = unidecode(soup.title.string).split()
        if temp[-1] in ['m4m', 'm4w', 'w4m', 'w4w']:
            category = temp.pop()
            title = ' '.join(temp[:-1])
        else:
            category = None
            title = unidecode(soup.title.string)
        return title, category

    def _get_datetime(self, soup):
        datetime_texts= soup.findAll('p', {'class': 'postinginfo reveal'})
        # If post has an "updated" timestamp, store that one
        datetimes = [item.time.text for item in datetime_texts]
        max_dt = max(datetimes)
        dt = pd.to_datetime(max_dt)
        return dt

    def _get_coords(self, soup):
        latlong = soup.find('div', {'class': 'viewposting'})
        if latlong:
            latitude = float(latlong['data-latitude'])
            longitude = float(latlong['data-longitude'])
            return (latitude, longitude)
        return (None, None)

    def _get_nhood(self, soup):
        neighborhood = soup.find('small')
        if neighborhood:
            neighborhood = str(unidecode(neighborhood.text)).strip('( )')
            return neighborhood
            # if neighborhood in self._nhoods:
            #     return neighborhood
        return None

    def _get_extra_age(self, soup):
        extra = soup.findAll('p', {'class': 'attrgroup'})
        if extra:
            extra = ' '.join([info.text.encode('utf-8') for info in extra])
            ind = extra.find('age: ')
            if ind >= 0:
                age = int(extra[ind+5:])
            else:
                age = -1
        else:
            age = -1
        return extra, age

    def _get_post(self, soup):
        text = []
        contents = soup.find(id='postingbody').contents
        for item in contents:
            str_item = str(unidecode(item))
            str_item = html_to_text(str_item)
            text.extend(str_item.strip().split())
        post = ' '.join(text)
        phrase = " it's NOT ok to contact this poster with services or other commercial interests"
        post.replace(phrase, '')
        post.replace('[?]', '', 200) # gets rid of emoji placeholders
        if ' Location: ' in post:
            end_ind = post.index(' Location: ')
            post = post[:end_ind]
        if ' <!-- ' in post:
            end_ind = post.index(' <!-- ')
            post = post[:end_ind]
        return post

    def _get_hrefs(self, soup):
        tags = soup.findAll('a',  {'class': 'hdrlnk'})
        hrefs = [str(tag['href']) for tag in tags if not str(tag['href']).startswith('//')]
        return hrefs

    def _has_pic(self, soup):
        if soup.findAll('figure'):
            return 1
        return 0

    def _look_at_page(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        return soup

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = [ ]

    def handle_data(self, d):
        self.result.append(d)

    def handle_charref(self, number):
        codepoint = int(number[1:], 16) if number[0] in (u'x', u'X') else int(number)
        self.result.append(unichr(codepoint))

    def handle_entityref(self, name):
        self.result.append('&%s;' % name)

    def get_text(self):
        return u''.join(self.result)

def html_to_text(html):
    s = HTMLTextExtractor()
    s.feed(html)
    return s.get_text()
