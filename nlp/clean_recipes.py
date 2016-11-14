from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import cPickle as pickle
import numpy as np
import pandas as pd
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
        self.unique_recipes = set([_['recipe_id'] for _ in RECIPE_COLLECTION.find({}, {'recipe_id': 1, '_id': 0})])

    def _stop_words(self):
        recipe_stopwords = set(['pound', 'pounds', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'cup', 'cups', 'bunch', 'chopped', 'diced', 'crushed', 'inch', 'sliced', 'optional', 'desired', 'ounce', 'ounces', 'fresh', 'piece', 'pinch', 'sprinkling', 'peeled', 'taste', 'quartered', 'halved', 'half', 'divided', 'lengthwise', 'box', 'package', 'packaged', 'uncooked', 'cooked', 'seared', 'drained', 'trimmed', 'mashed', 'grated', 'ground', 'shredded', 'cut', 'cube', 'cubed', 'prepared', 'fresh', 'freshly', 'dried', 'fresh', 'beaten', 'lightly', 'light', 'room', 'temperature', 'skinless', 'boneless', 'half', 'chunk', 'yummy', 'snipped', 'fillet', 'whole', 'husk', 'removed', 'thin', 'thinly', 'thickly', 'thick', 'soft', 'large', 'ripe', 'large', 'pressed', 'jar', 'rinsed', 'well', 'dash', 'can', 'melted', 'thawed', 'softened', 'degree', 'degrees', 'c', 'finely', 'bag', 'bags', 'baby', '16inch', '10inch', '12inch', '14inch', '18inch', '3inch', '4inch', '5inch', '6inch', '7inch', '9inch', '1inch', 'unpeeled', 'peeled', 'with', 'without', '1pint', '1quart', '2layer', ' '])
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

    def fit_vectorizor(self, pkl=False):
        recipe_cursor = RECIPE_COLLECTION.find(no_cursor_timeout=True)#.limit(100)
        for recipe in recipe_cursor:
            text = recipe['ingredients']
            # try:
            #     text.extend(recipe['taxonomy'])
            # except KeyError:
            #     continue
            for word in recipe['name'].split('-'):
                text.append(word)
            self.documents.append(self._tokenize(text))

        self.tfidf = self.vectorizer.fit_transform(self.documents)
        if pkl:
            with open("nlp_vectorizer.pkl", 'w') as f:
                pickle.dump(self.vectorizer, f)

    def get_recipe_text(self, recipe_id):
        recipe = RECIPE_COLLECTION.find_one({'recipe_id': str(recipe_id)})
        if recipe:
            text = recipe['ingredients']
            # try:
            #     text.extend(recipe['taxonomy'])
            # except KeyError:
            #     next
            for word in recipe['name'].split('-'):
                text.append(word)
            return text

    def recipe_text_to_df(self, ids, filename=None, from_pickle=False):
        if from_pickle:
            path = '/Users/Gavin/ds/recipe_recommender/nlp/'
            with open(path + filename) as f_un:
                vect = pickle.load(f_un)
        else:
            self.fit_vectorizor()
            vect = self.vectorizer

        document_text_matrices= []
        for recipe_id in ids:
            mat = vect.transform(self.get_recipe_text(recipe_id))
            document_text_matrices.append(mat.toarray())
        df = pd.DataFrame(document_text_matrices)
        return df

    def test(self, num_recipes=1):
        recipe_ids = []
        for recipe in RECIPE_COLLECTION.find().limit(num_recipes):
            recipe_ids.append(recipe['recipe_id'])
            text = recipe['ingredients']
            text.extend(recipe['taxonomy'])
            text.append(' '.join(recipe['name'].split('-')))
            documents = self._tokenize(text)
        return text, documents, recipe_ids
    #
    def df_of_text(self):
        for recipe in self.unique_recipes:
            tokens = self._tokenize(self.get_recipe_text(recipe))

    def taxonomy_to_df(self):
        tax_lst, recipe_ids= [], []
        for recipe in RECIPE_COLLECTION.find({}, {'taxonomy': 1, 'recipe_id': 1, '_id': 0}):
            try:
                recipe_ids.append(recipe['recipe_id'])
                tax_lst.append(recipe['taxonomy'][0])
            except KeyError:
                recipe_ids.append(recipe['recipe_id'])
                tax_lst.append(np.nan)
        df = pd.DataFrame(np.array(zip(recipe_ids, tax_lst)))
        df.columns =['recipe_id', 'taxonomy']

        groups = df['taxonomy'].unique().tolist()
        d = dict(zip(groups, range(len(groups))))
        df['taxonomy_id'] = df['taxonomy'].map(d)
        df.drop('taxonomy', axis=1, inplace=True)
        return df


# def update_recipes():
#     for recipe in COLLECTION.find():
#         COLLECTION.update_one({'_id': recipe['_id']},
#                 {'$set': {'tokens': clean_ingredients(recipe['ingredients'])}},
#                 upsert=False)
#         # print clean_ingredients(recipe['ingredients'])
#     print 'Updated!'

#
# if __name__ == '__main__':
#     # t = clean_ingredients(test_list)
#     # update_recipes()
#     # text, documetns, recipe_ids = t.test(num_recipes=10)
#     test_ids = [u'212940',
#                  u'230469',
#                  u'50026',
#                  u'219164',
#                  u'255821',
#                  u'246866',
#                  u'255298',
#                  u'13010',
#                  u'237093',
#                  u'231067']
#     t = NLPProcessor()
#     # t.make_tfidf(test_ids)
#     # t.vectorizer.vocabulary_
#     t.fit_vectorizor()
#     text = t._tokenize(t.get_recipe_text('212940'))
#     # test = t.recipe_text_to_df(test_ids)
#     # test = t.taxonomy_to_df()
#
#
#     '''
#     getting nlp to work
#
#     enter taxonomy in one column and cluster group in another one.
#     '''
#
#
#     '''
#     send vectorizer vocab to ordered dict
#
#     loop through ordered dict and if word from individual recipe is in ordered dict then append 1 otherwise append 0 to a deque
#
#     add to entry in mongo for speed of referencing
#
#     when pull into sframe when called
#     '''
