from bs4 import BeautifulSoup
import urllib, urllib2
from pymongo import MongoClient
from collections import defaultdict
from unidecode import unidecode
import requests

#global variables
DB_NAME = 'allrecipes'
# COLL_NAME = 'recipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
# COLLECTION = DATABASE[COLL_NAME]

def load_recipe_ids():
    '''
    Input: None.
    Output: None.
    loads recipe_ids from MongoDB. Sends to find raters.

    Takes user_ids from
    '''
    recipe_collection = DATABASE['recipes']
    for recipe in recipe_collection.find():
        find_rater(recipe['recipe_id'])

def find_raters(recipe_id):
    '''
    Input: string (recipe_id)
    Output: dictionary (unique user_ids)

    Finds raters of the recipes from recipe_id. Returns dictionary with key as user_id and values as list of dictionaries where each key is the recipe_id and value is user rating.
    '''
    recipe_url = 'http://allrecipes.com/recipe/' + recipe_id
    r = urllib2.urlopen(recipe_url)
    soup = BeautifulSoup(r.read(), 'html.parser')

    for review in soup.findAll('div', {'class': 'review-container clearfix'}):
        entry = defaultdict(list)
        rater_id = unidecode(review.find('a')['href']).split('/')[-2]
        rating = review.find('div', {'class': 'rating-stars'})['data-ratingstars']

        entry[rater_id].append({recipe_id: int(rating)})
        find_user_preferences(entry)
    print 'Shiat be entered'

def find_user_preferences(entry):
    '''
    Input: dictionary
    Output: None
    '''
    #set Mongo collection to
    user_collection = DATABASE['users']
    user_id = entry.keys()[0]
    header = {
        'Host': 'apps.allrecipes.com',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
        'Origin': 'http://allrecipes.com',
        'X-Requested-With': 'XMLHttpRequest',
        'Authorization': 'Bearer P6k/U+2F1ECWIwpmI527pX44t7ijF2Lgfk7Ve/QnsHjEVSm4Anh5KiFhE+WTsIsvHyT7DBZHiGEtt23WU7GkfO3fBTzcT9zInDsp8Z9Hzc0iCPtGC4znb6CZGd3lbZpl',
        'Accept': '*/*',
        'Referer': 'http://allrecipes.com/cook/' + user_id + '/reviews/',
        'Accept-Encoding': 'gzip, deflate, sdch, br',
        'Accept-Language': 'en-US,en;q=0.8'
        }


    payload1 = {
            'page': 1,
            'pageSize':100,
            'tenantId':12,
            }
    request_url = 'https://apps.allrecipes.com/v1/users/'+ user_id + '/reviews'
    r = requests.get(request_url, params=payload1, headers=header)

    if r.status_code == 200:
        r = r.json()
        # num_pages = len(r['links'])

        for review in r['reviews']:
            rating = review['rating'] #r.json()['reviews'][0]['rating']
            recipe_id = review['recipe']['recipeID'] #r.json()['reviews'][0]['recipe']['recipeID']

            entry[user_id].append({str(recipe_id): int(rating)})
        user_collection.insert_one(entry)
    else:
        print 'Error. status_code: ', r.status_code


    # if len(r['links']) > 1:
    #     for page in range(2, len)
    #     payload2 = {
    #                 'page': 1,
    #                 'pageSize':20,
    #                 'tenantId':12,
    #                 }


    return r

if __name__ == '__main__':
    recipe_id = '24243'
    cook_id = '461332'
    test_entry = {"461332" : [ { "24243" : 5 } ] }
    test_entry1 = {"209823" : [ { "24243" : 5 } ] }
    # find_raters(recipe_id)
    test = find_user_preferences(test_entry1)
