import urllib, urllib2
import time
import json
import requests
from pymongo import MongoClient

#global variables
DB_NAME = 'allrecipes'
COLL_NAME = 'deals'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
COLLECTION = DATABASE[COLL_NAME]

class IbottaScraper(object):
    '''
    class to get grocery offer data from Ibotta.com. Processes non-duplicated offers to MongoDB.

    To get App_Token:
        1. go to ibotta.com and filter by grocery
        2. With google dev tools
            a. right click -> inspect -> network -> XHR -> refresh page
            b. click to open details of 'offer.json'
            c. scroll to 'request-headers' section. replace 'X-App-Token' of class 'self.header' with new values in 'request-headers' section.
    '''
    def __init__(self, App_Token):
        self.time = time.time()
        #url to .json file which is loaded via ajax
        self.url = 'https://ibotta.com/web_v1/offers.json'
        #headers needed to complete GET request
        self.header = {
            'Host': 'ibotta.com',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-App-Version': '4.9.4:webapp',
            'X-Requested-With': 'XMLHttpRequest',
            'X-App-Token': App_Token,
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
            'Referer': 'https://ibotta.com/rebates?category=grocery',
            'Accept-Encoding': 'gzip, deflate, sdch, br',
            'Accept-Language': 'en-US,en;q=0.8',
            'Cookie': '__ssid=4ddda54d-af1a-4883-8f3d-648adc887fe3; viewedOuibounceModal=true; _ga=GA1.2.2058757665.1472082122; _gat=1; ab.storage.sessionId.cb535ba1-2856-474b-9f5d-1416c922440d=%7B%22g%22%3A%22511b896c-b4fd-a476-7de2-97fce85e79cd%22%2C%22e%22%3A1477944569441%2C%22c%22%3A1477942769442%2C%22l%22%3A1477942769442%7D',
            'If-Modified-Since': 'Mon, 31 Oct 2016 5:00:00 GMT',
            }
        #categories that are of interest for recipes
        self.offer_categories = {
            'Pantry' : 450,
            'Frozen' : 13,
            'Beverages' : 1,
            'Snacks' : 449,
            'Mixers' : 258,
            'Dairy & Eggs' : 12,
            'Wine' : 171,
            'Spirits' : 231,
            'Candy & Sweets' : 416,
            'Health & Wellness' : 344,
            'Meals & Beverages' : 285,
            'Breakfast' : 9,
            'Flavored Alcoholic Beverages' : 482,
            'Beer & Cider' : 172,
            'Meat & Seafood' : 422,
            'Bread & Bakery' : 415,
            }
        self._scrape()

    def _scrape(self):
        r = requests.get(self.url,
            headers=self.header)
        if r.status_code == 200 or r.status_code == 304: #304 means data not modified
            print 'Success! status code', r.status_code
            self.data = json.loads(r.text)
            self._process()
        else:
            print 'Error: status code ', r.status_code

    def _process(self):
        for offer in self.data['offers']:
            for cat_id in offer['offer_category_ids']:
                if cat_id in self.offer_categories.values():
                    if not bool(COLLECTION.find({'id': offer['id']}).count()):
                        # offer['tags'] = [t['offer_tag']for t in offer['offer_tags']]
                        COLLECTION.insert_one(offer)

if __name__ == '__main__':
    token = 'lyWreHoghkuWyed6:1478024364:d325cabfe1dff105eaf87311111fdfb4b90411dd'
    #App_Tokens expire when browser session is closed so need to get new one at least daily
    ibotta = IbottaScraper(App_Token=token)
