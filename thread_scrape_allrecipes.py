from bs4 import BeautifulSoup
import urllib, urllib2
from pymongo import MongoClient
import multiprocessing
#bring in custom class for threading
from allrecipes_threader import RecipeThreader


#global variables
DB_NAME = 'testdb'
COLL_NAME = 'test'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
COLLECTION = DATABASE[COLL_NAME]


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
            href = dish.find('a')['href']
            if href[:7] == '/recipe': #handles hrefs that are videos/other
                recipe_links.append(href)
        except TypeError: #Error handling for advertisements
            pass
    return recipe_links

def recipe_data_to_mongo(max_page):
    for i in xrange(1, max_page+1):
        links = get_recipes_by_page_number(i)
        for link in links:
            #verify recipe does not already exist in database
            recipe_id = link.split('/')[2]
            if not bool(COLLECTION.find({'recipe_id': recipe_id}).count()):
                worker = RecipeThreader(link)
                worker.start()
                workers.append(worker)

        workers = []
        for worker in workers:
            worker.join()
            print worker.entry, worker.link
            COLLECTION.insert_one(worker.entry)
        print 'Success!'


if __name__ == '__main__':
    max_page = 2
    recipe_data_to_mongo(max_page)
    # test_class = RecipeThreader('/recipe/8778/cajun-chicken-pasta/')
