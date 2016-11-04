from bs4 import BeautifulSoup
import urllib, urllib2
import requests
from collections import defaultdict
from threading import Thread
from unidecode import unidecode

class RecipeThreader(Thread):
    '''
    Class inherits from threading.Thread. Allows storing of scraped values in dictionary which can then be referenced after the thread has stopped.

    Convenient for putting scraped data in a MongoDB for example.
    '''

    def __init__(self, link):
        #self variable related run function (recipe related)
        self.link = link
        self.entry = defaultdict(list)
        response = urllib2.urlopen('http://allrecipes.com'+ self.link)
        html = response.read()
        self.soup = BeautifulSoup(html, 'html.parser')

        #self variable relate get_users fuction
        self.recipe_id = self.link.split('/')[2]
        self.request_url = 'http://allrecipes.com/recipe/getreviews/'
        self.user_set = set()
        self.header = { 'Host': 'allrecipes.com',
                    'Connection': 'keep-alive',
                    'X-NewRelic-ID': 'XQ4OVVdaGwADVlhaBQcF',
                    'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                    'Accept': '*/*',
                    'Referer': 'http://allrecipes.com/recipe/' + self.recipe_id,
                    'Accept-Encoding': 'gzip, deflate, sdch',
                    'Accept-Language': 'en-US,en;q=0.8',
                    }
        #other
        super(RecipeThreader, self).__init__()

    def run(self):
        #name
        self.entry['name'] = self.link.split('/')[-2]
        self.entry['recipe_id'] = self.link.split('/')[2]
        #categories
        for category in self.soup.findAll('span', attrs={'class' : "toggle-similar__title"})[2:]:
            self.entry['taxonomy'].append(category.text.strip())
        #ingredients
        for ingredient in self.soup.findAll('span', attrs={'class': 'recipe-ingred_txt added'}):
            self.entry['ingredients'].append(ingredient.text)
        #pic url
        self.entry['picture'].append(self.soup.findAll('img', attrs={'class', 'rec-photo'})[0]['src'])
        #directions
        for step in self.soup.findAll('span', attrs={'class': 'recipe-directions__list--item'}):
            self.entry['directions'].append(step.text)
        #prep, cook, and total time
        for item in self.soup.findAll('time'):
            self.entry['time'].append(item.text)
        #reviews
        self.entry['rating'].append(self.soup.find('div', attrs={'class', 'rating-stars'})['data-ratingstars'])
        self.entry['num_reviews'].append(int(self.soup.find('meta', attrs={'itemprop': 'reviewCount'})['content']))
        #call _get_users()
        # self._get_users()

    def _get_users(self):
        '''
        gets user_ids of users that reviewed the recipe.
        '''
        review_types = ['MostPositive', 'LeastPositive', 'Newest', 'MostHelpful']
        for group in review_types:
            payload = {'recipeid': self.recipe_id,
                        'pagenumber': 1,
                        'pagesize': 100,
                        'recipeType': 'Recipe',
                        'sortBy': group}

            r = requests.get(self.request_url, params=payload, headers=self.header)
            soup = BeautifulSoup(r.text, 'html.parser')
            for tag in soup.findAll('a'):
                split = tag['href'].split('/')
                if split[1] == 'cook' and split[2].isdigit():
                    user_id = split[2].strip()
                    self.user_set.add(unidecode(user_id))

class UserThreader(Thread):
    '''
    Class inherits from threading.Thread. Allows storing of scraped values in dictionary which can then be referenced after the thread has stopped.

    Convenient for putting scraped data in a MongoDB for example.
    '''

    def __init__(self, user_id):
        self.user_id = user_id
        self.entry = {'user_id': self.user_id, 'ratings': []}
        self.header = {
            'Host': 'apps.allrecipes.com',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
            'Origin': 'http://allrecipes.com',
            'X-Requested-With': 'XMLHttpRequest',
            'Authorization': 'Bearer P6k/U+2F1ECWIwpmI527pX44t7ijF2Lgfk7Ve/QnsHjEVSm4Anh5KiFhE+WTsIsvHyT7DBZHiGEtt23WU7GkfO3fBTzcT9zInDsp8Z9Hzc0iCPtGC4znb6CZGd3lbZpl',
            'Accept': '*/*',
            'Referer': 'http://allrecipes.com/cook/' + self.user_id + '/reviews/',
            'Accept-Encoding': 'gzip, deflate, sdch, br',
            'Accept-Language': 'en-US,en;q=0.8'
            }
        self.payload = {
                'page': 1,
                'pageSize':100, #set page size equal to 100 reviews believing it is very unlikely a user has rated more than 100 recipes.
                'tenantId':12,
                }
        super(UserThreader, self).__init__()

    def run(self):
        request_url = 'https://apps.allrecipes.com/v1/users/'+ self.user_id + '/reviews'
        r = requests.get(request_url, params=self.payload, headers=self.header)
        if r.status_code == 200:
            r = r.json()
            for review in r['reviews']:
                rating = review['rating']
                recipe_id = review['recipe']['recipeID']
                self.entry['ratings'].append({str(recipe_id): int(rating)})
        else:
            print 'Error. status_code: ', r.status_code

if __name__ == '__main__':
    test = UserThreader('461332')
    test.run()
    # pass
