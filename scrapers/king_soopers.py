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

def scrape_king_soopers(to_database=True):
    url = 'https://kroger.softcoin.com/p/np/4230/Kroger/coupons'
    payload = {
                'banner': 'King_Soopers',
                'usource': 'KWL%7Cm',
                '_': 1478465892420
                }
    header = {
            'Host': 'kroger.softcoin.com',
            'Connection': 'keep-alive',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
            'Referer': 'https://kroger.softcoin.com/programs/kroger/digital_coupons/?origin=DigitalCoupons&banner=King_Soopers',
            'Accept-Encoding': 'gzip, deflate, sdch, br',
            'Accept-Language': 'en-US,en;q=0.8',
            'Cookie': 'JSESSIONID=a102~B257029188FD21B51BBC7CE6CB8B5643.a102; s_vnum=1481597951563%26vn%3D1; s_sq=%5B%5BB%5D%5D; s_fid=376FAA7EDA5F0F25-3B1D5E137D70B6A9; s_nr=1479006005284-Repeat; undefined_s=First%20Visit; s_invisit=true; s_cc=true'
            }
    r = requests.get(url, headers=header, params=payload)
    # print r.status_code, r.json()[0]
    coupons = r.json()['coupons']
    # deals = [c['categories'] for c in r.json()['coupons']]
    # categories = set([cat['category'] for deal in deals for cat in deal])
    if to_database:
        to_mongo(coupons)
    return coupons

def to_mongo(coupons):
    relevant_categories = {u'Bakery',u'Baking Goods',u'Beverages',u'Breakfast',u'Bulk Foods',u'Canned & Packaged',u'Condiment & Sauces',u'Dairy',u'Deli',u'Frozen',u'International',u'Meat & Seafood',u'Produce',u'Promotions',u'Snacks'}
    for coupon in coupons:
        if coupon['category'] in relevant_categories:
            if COLLECTION.find({"coupon_id" : coupon['coupon_id']}).count() == 0:
                coupon['source'] = 'King_Soopers'
                COLLECTION.insert_one(coupon)

def get_coupon_text():
    # l = ['display_description', 'short_description'
    for coupon in COLLECTION.find():
        text = ' '.join([coupon['display_description'], coupon["short_description"]])
        print text

if __name__ == '__main__':
    # token = 'lyWreHoghkuWyed6:1478024364:d325cabfe1dff105eaf87311111fdfb4b90411dd'
    #App_Tokens expire when browser session is closed so need to get new one at least daily
    # ibotta = IbottaScraper(App_Token=token)
    # t = scrape_king_soopers()
    get_coupon_text()
