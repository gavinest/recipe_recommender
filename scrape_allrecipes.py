import pandas as pd
import datetime
from bs4 import BeautifulSoup
import urllib, urllib2
import requests

from pymongo import MongoClient
from collections import defaultdict


def get_recipes_by_page_number(i):
    '''
    input: i, integer
    output: list

    takes page number input. returns list of urls for recipes on that page.
    '''

    url = 'http://allrecipes.com/recipes/?page={0}#{0}'.format(i)
    response = urllib2.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')

    dishes = soup.select('article.grid-col--fixed-tiles')
    recipe_links = []
    for dish in dishes:
        try:
            recipe_links.append(dish.find('a')['href'])
        except TypeError: #Error handling for advertisements
            pass
    return recipe_links

def recipe_data_to_mongo(i):
    db_client = MongoClient()
    db = db_client['allrecipes']
    collection = db['recipes']

    links = get_recipes_by_page_number(i)
    for link in links:
        entry = parse_recipe(link)
        collection.insert_one(entry)
    return True

def parse_recipe(link):
    gen_url = 'http://allrecipes.com'
    response = urllib2.urlopen(gen_url + link)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')

    entry = defaultdict(list)

    #name
    entry['name'] = link.split('/')[-2]

    #categories
    for category in soup.findAll('span', attrs={'class' : "toggle-similar__title"})[2:]:
        entry['taxonomy'].append(category.text)

    #ingredients
    for ingredient in soup.findAll('span', attrs={'class': 'recipe-ingred_txt added'}):
        entry['ingredients'].append(ingredient.text)

    #pic url
    entry['picture'].append(soup.findAll('img', attrs={'class', 'rec-photo'})[0]['src'])

    #directions
    for step in soup.findAll('span', attrs={'class': 'recipe-directions__list--item'}):
        entry['directions'].append(step.text)

    #prep, cook, and total time
    for item in soup.findAll('time'):
        entry['time'].append(item.text)

    #reviews
    entry['rating'].append(soup.find('div', attrs={'class', 'rating-stars'})['data-ratingstars'])

    return entry


def main(i):
    if recipe_data_to_mongo(i):
        print 'Success!'

if __name__ == '__main__':
    # main(1)
    # test_link = '/recipe/24272/buttery-soft-pretzels/'
    # entry = parse_recipe(test_link)
    main(1)
