from bs4 import BeautifulSoup
import urllib, urllib2
from pymongo import MongoClient
from collections import defaultdict
from unidecode import unidecode
import requests


#global variables
# DB_NAME = 'allrecipes'
# CLIENT = MongoClient()
# DATABASE = CLIENT[DB_NAME]
# RECIPE_COLLECTION = DATABASE['recipes']
# USER_COLLECTION = DATABASE['users']
#
# class UserThreader(Thread):
#     '''
#     Class inherits from threading.Thread. Allows storing of scraped values in dictionary which can then be referenced after the thread has stopped.
#
#     Convenient for putting scraped data in a MongoDB for example.
#     '''
#
#     def __init__(self, link):
#         self.link = link
#         self.entry = defaultdict(list)
#         response = urllib2.urlopen('http://allrecipes.com'+ self.link)
#         html = response.read()
#         self.soup = BeautifulSoup(html, 'html.parser')
#         super(RecipeThreader, self).__init__()

def find_user_ratings(entry):
    '''
    Input: dictionary
    Output: None
    '''
    #set Mongo collection to
    user_id = entry['user_id']
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
            'pageSize':100, #set page size equal to 100 reviews believing it is very unlikely a user has rated more than 100 recipes.
            'tenantId':12,
            }

    request_url = 'https://apps.allrecipes.com/v1/users/'+ user_id + '/reviews'
    r = requests.get(request_url, params=payload1, headers=header)

    if r.status_code == 200:
        r = r.json()
        for review in r['reviews']:
            rating = review['rating']
            recipe_id = review['recipe']['recipeID']
            if bool(RECIPE_COLLECTION.find({'recipe_id': recipe_id}).count()):
                entry['ratings'].append({str(recipe_id): int(rating)})
        USER_COLLECTION.insert_one(entry)
    else:
        print 'Error. status_code: ', r.status_code

# def load_recipe_ids():
#     '''
#     Input: None.
#     Output: None.
#     loads recipe_ids from MongoDB. Sends to find raters.
#
#     Takes user_ids from
#     '''
#     for recipe in RECIPE_COLLECTION.find():
#         find_raters(recipe['recipe_id'])

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
        print review
    #     entry = {}
    #     user_id = unidecode(review.find('a')['href']).split('/')[-2]
    #     if not bool(USER_COLLECTION.find({'user_id': user_id}).count()):
    #         # rating = review.find('div', {'class': 'rating-stars'})['data-ratingstars']
    #         entry['user_id'] = user_id
    #         entry['ratings'] = []
    #         find_user_ratings(entry)
    # print 'Success!'
    return soup

def find_raters_many(recipe_id):
    request_url = 'http://allrecipes.com/recipe/getreviews/'
    #'?recipeid=24243&pagenumber=1&pagesize=9&recipeType=Recipe&sortBy=MostPositive'

    header = { 'Host': 'allrecipes.com',
                'Connection': 'keep-alive',
                'X-NewRelic-ID': 'XQ4OVVdaGwADVlhaBQcF',
                'X-Requested-With': 'XMLHttpRequest',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                'Accept': '*/*',
                'Referer': 'http://allrecipes.com/recipe/' + recipe_id,
                'Accept-Encoding': 'gzip, deflate, sdch',
                'Accept-Language': 'en-US,en;q=0.8',
                }

    review_types = ['MostPositive', 'LeastPositive', 'Newest', 'MostHelpful']
    user_set = set()
    for group in review_types:
        payload = {'recipeid': recipe_id,
                    'pagenumber': 1,
                    'pagesize': 100,
                    'recipeType': 'Recipe',
                    'sortBy': group}

        r = requests.get(request_url, params=payload, headers=header)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup.findAll('a'):
            split = tag['href'].split('/')
            if split[1] == 'cook' and split[2].isdigit():
                user_id = split[2]
                user_set.add(unidecode(user_id))   
    return user_set

def make_soup(url):
    response = urllib2.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup

if __name__ == '__main__':
    recipe_id = '24243'
    cook_id = '461332'
    test_entry = {'user_id': '461332', 'ratings': []}
    # find_raters(recipe_id)
    url = 'http://allrecipes.com/recipe/245361/creamy-pork-stew/'
    # soup = make_soup(url)
    # test = find_user_ratings(test_entry)
    test = find_raters_many(recipe_id)


    '''
    div class="profile profile--recipe-card">\n<a data-click-id="cardslot 3" data-internal-referrer-link="rr_recipe_a" href="/cook/10642/">\n<ul class="cook-details">\n<li>\n<img alt="profile image" class="img-profile elevate-cook-thumbnail" src="http://images.media-allrecipes.com/global/features/mini/1747.jpg"/>\n</li>\n<li>\n<h4><span>Recipe by</span> Michele O'Sullivan</h4>

    '''
