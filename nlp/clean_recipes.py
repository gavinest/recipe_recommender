from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader

#GLOBAL VARIABLES
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

class NLPProcessor(object):
    '''
    Inputs: Vectorizer type - default=Tfidf
            kwargs - default to list analyzer for allowing n_grams and lowercase as False for speed
            # kwargs - additional keyword arguments. i.e. max_df, min_df

    Class performs text cleaning [extract stopwords, lemmatization] of recipes in MonogDB. Makes tfidf of documents in question
    '''
    def __init__(self, vectorizer=TfidfVectorizer, kwargs={'analyzer':list, 'lowercase':False}):
        self.documents = []
        #create vectorizer expecting list of input for each document to allow for n-grams
        #set lowercase to False since text already lowercased
        self.vectorizer = vectorizer(**kwargs)

    def _stop_words(self):
        recipe_stopwords = set(['pound', 'pounds', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'cup', 'cups', 'bunch', 'chopped', 'diced', 'crushed', 'inch', 'sliced', 'optional', 'desired', 'ounce', 'ounces', 'fresh', 'piece', 'pinch', 'sprinkling', 'peeled', 'taste', 'quartered', 'halved', 'half', 'divided', 'lengthwise', 'box', 'package', 'packaged', 'uncooked', 'cooked', 'seared', 'drained', 'trimmed', 'mashed', 'grated', 'ground', 'shredded', 'cut', 'cube', 'cubed', 'prepared', 'fresh', 'freshly', 'dried', 'fresh', 'beaten', 'lightly', 'light', 'room', 'temperature', 'skinless', 'boneless', 'half', 'chunk', 'yummy', 'snipped', 'fillet', 'whole', 'husk', 'removed', 'thin', 'thinly', 'thickly', 'thick', 'soft', 'large', 'ripe', 'large', 'pressed', 'jar', 'rinsed', 'well', 'dash', 'can'])
        stop = set(stopwords.words('english'))
        return stop.union(recipe_stopwords)

    def _tokenize(self, text):
        stop = self._stop_words()
        tokens = []
        for line in text:
            line = [_ for _ in unidecode(line).lower().translate(None, punctuation).split() if not _.isdigit()]
            lemmers = [WordNetLemmatizer().lemmatize(word) for word in line if word not in stop]
            tokens.append(' '.join(lemmers))
        return tokens

    def make_tfidf(self, ids):
        for recipe_id in ids:
            recipe = RECIPE_COLLECTION.find_one({'recipe_id': str(recipe_id)})
            if recipe:
                text = recipe['ingredients']
                text.extend(recipe['taxonomy'])
                for word in recipe['name'].split('-'):
                    text.append(word)
                self.documents.append(self._tokenize(text))
            else:
                self.documents.append(' ')

        self.tfidf = self.vectorizer.fit_transform(self.documents)

    def test(self, num_recipes=1):
        recipe_ids = []
        for recipe in RECIPE_COLLECTION.find().limit(num_recipes):
            recipe_ids.append(recipe['recipe_id'])
            text = recipe['ingredients']
            text.extend(recipe['taxonomy'])
            text.append(' '.join(recipe['name'].split('-')))
            documents = self._tokenize(text)
        return text, documents, recipe_ids

# def update_recipes():
#     for recipe in COLLECTION.find():
#         COLLECTION.update_one({'_id': recipe['_id']},
#                 {'$set': {'tokens': clean_ingredients(recipe['ingredients'])}},
#                 upsert=False)
#         # print clean_ingredients(recipe['ingredients'])
#     print 'Updated!'


if __name__ == '__main__':
    # t = clean_ingredients(test_list)
    # update_recipes()
    t = NLPProcessor()
    # text, documetns, recipe_ids = t.test(num_recipes=10)
    test_ids = [u'212940',
                 u'230469',
                 u'50026',
                 u'219164',
                 u'255821',
                 u'246866',
                 u'255298',
                 u'13010',
                 u'237093',
                 u'231067']
    t.get_tfidf(test_ids)
    t.vectorizer.vocabulary_
