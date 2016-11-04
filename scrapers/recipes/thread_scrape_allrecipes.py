from bs4 import BeautifulSoup
import urllib, urllib2
from pymongo import MongoClient
import multiprocessing
#bring in custom class for threading
from allrecipes_threader import RecipeThreader, UserThreader

#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

def parallelize(max_page):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.map(func=recipe_data_to_mongo, iterable=xrange(1, max_page+1))

def recipe_data_to_mongo(i):
    links = get_recipes_by_page_number(i)
    workers = []
    for link in links:
        #verify recipe does not already exist in database
        recipe_id = link.split('/')[2]
        if RECIPE_COLLECTION.find({'recipe_id': recipe_id}).count() == 0:
            #create thread workers for each recipe on page
            worker = RecipeThreader(link)
            worker.start()
            workers.append(worker)

    for worker in workers:
        worker.join()
        RECIPE_COLLECTION.insert_one(worker.entry)
        # user_data_to_mongo(worker.user_set) #add list of users that reviewed recipe to set
    print 'Page {} Success!'.format(i)


def user_data_to_mongo(users):
    workers = []
    for user in users:
        if USER_COLLECTION.find({'user_id': user}).count() == 0:
            #create thread workers for each user
            worker = UserThreader(user)
            worker.start()
            workers.append(worker)

    for worker in workers:
        worker.join()
        USER_COLLECTION.insert_one(worker.entry)
        # print 'User {} Added!'.format(worker.user_id)

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

if __name__ == '__main__':
    max_page = 10000
    # # recipe_data_to_mongo(max_page)
    parallelize(max_page)
