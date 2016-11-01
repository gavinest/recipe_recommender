from bs4 import BeautifulSoup
import urllib, urllib2
from collections import defaultdict
from threading import Thread


class RecipeThreader(Thread):
    '''
    Class inherits from threading.Thread. Allows storing of scraped values in dictionary which can then be referenced after the thread has stopped.

    Convenient for putting scraped data in a MongoDB for example.
    '''

    def __init__(self, link):
        self.link = link
        self.entry = defaultdict(list)
        response = urllib2.urlopen('http://allrecipes.com'+ self.link)
        html = response.read()
        self.soup = BeautifulSoup(html, 'html.parser')
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
