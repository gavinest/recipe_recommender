import numpy as np
import pandas as pd
from unidecode import unidecode
from pymongo import MongoClient
from scipy import sparse
import time
from collections import deque

#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

class DataLoader(object):
    '''
    Input: int. (number of users to include in data table)

    Reads user and recipe data from MonogDB. Allowing outputs of various data types.
        - sparse matrix
        - pandas dataframe
        - pickle file
    '''
    def __init__(self, n_users=None):
        self.n_users = n_users
        self.recipe_idx = {}
        self.user_idx = {}
        self.recipe_ids = deque()
        self.user_ids = deque()
        self.all_recipe_ids = set()

    def to_matrix(self):
        self.n_recipes = RECIPE_COLLECTION.find().count()
        self.ratings_dictionary = {}

        #recipes to numpy array and make dictionary of recipe_ids with index in matrix
        for r_idx, recipe in enumerate(RECIPE_COLLECTION.find(), start=0):
            recipe_id = recipe['recipe_id']
            self.recipe_idx[recipe_id] = r_idx
            self.recipe_ids.append(recipe_id)

        user_cursor = USER_COLLECTION.find(no_cursor_timeout=True).limit(self.n_users)
        for u_idx, user in enumerate(user_cursor, start=0): #
            user_id = user['user_id']
            self.user_idx[user_id] = u_idx
            self.user_ids.append(user_id)
            for review in user['ratings']:
                self.all_recipe_ids.add(review.keys()[0])
                if review.keys()[0] in self.recipe_idx.keys():
                    r_idx = self.recipe_idx[review.keys()[0]]
                    self.ratings_dictionary[(u_idx, r_idx)] = review.values()[0]
        user_cursor.close()

        self.sparse_mat = sparse.dok_matrix((self.n_users, self.n_recipes))
        for idx, rating in self.ratings_dictionary.iteritems():
            self.sparse_mat[idx] = rating

        self.sparse_mat.transpose().tocsr()

    def to_dataframe(self):
        self.to_matrix()
        df = pd.DataFrame(self.sparse_mat.toarray(), index=self.user_ids, columns=self.recipe_ids, dtype='int64')
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=['index'], value_vars=df.columns.tolist()[1:])
        df.columns = ['user_id', 'recipe_id', 'rating']
        self.df = df[df['rating'] != 0]

    def to_pickle(self, filename):
        '''
        input: string
        '''
        path = '/Users/Gavin/ds/recipe_recommender/data_management/'
        self.to_dataframe()
        self.df.to_pickle(path + filename)

    def get_one_user(self, user_id):
        reviews = USER_COLLECTION.find_one({'user_id': user_id})['ratings']
        n = len(reviews)
        recipe_list, rating_list = zip(*[[review.keys()[0], review.values()[0]] for review in reviews])
        df = pd.DataFrame({'user_id': [user_id]*n, 'recipe_id': recipe_list, 'rating': rating_list})
        return reviews

    def pull_recipe_data(self, ids):
        '''
        Input: List
        Output: List

        Takes list of recipe ids. Querires MongoDB. Returns list of full recipe cards.
        '''
        recipe_cards = []
        for recipe_id in ids:
            recipe_cards.append(RECIPE_COLLECTION.find_one({'recipe_id': str(recipe_id)}))
        return recipe_cards

    def add_recipes(self, ids):
        '''
        input: list or recipe_ids
        '''
        print '{} Total recipes to get.'.format(len(ids))
        for recipe_id in ids:
            link = '/recipe/' + str(recipe_id)
            r = RecipeThreader(link)
            RECIPE_COLLECTION.insert_one(worker.entry)
        print 'all recipes added!'

    def avg_rating_df(self):
        avg_ratings = []
        for recipe in RECIPE_COLLECTION.find({}, {'recipe_id':1, 'rating': 1, 'num_reviews': 1, '_id':0}):
            avg_ratings.append([recipe['recipe_id'], recipe['rating'][0], recipe['num_reviews'][0]])
        df = pd.DataFrame(np.array(avg_ratings), columns=['recipe_id', 'avg_rating', 'num_reviews'], dtype='float64')
        df['recipe_id'] = df['recipe_id'].astype('int64')
        df['num_reviews'] = df['num_reviews'].astype('int64')
        return df

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

if __name__ == '__main__':
    # start = time.time()
    #
    # d = DataLoader(100)
    # d.to_pickle('data_1H.pkl')
    # # d.to_dataframe()
    #
    # total_time = time.time()-start
    # print total_time

    # df = pd.read_pickle('data.pkl')
    t = DataLoader()
    avg_df = t.avg_rating_df()
    tax_df = t.taxonomy_to_df()
