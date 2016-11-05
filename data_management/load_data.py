import numpy as np
import pandas as pd
from unidecode import unidecode
from pymongo import MongoClient
from scipy import sparse

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
    '''
    def __init__(self, n_users=None):
        self.n_users = n_users
        self.recipe_idx = {}
        self.user_idx = {}
        self._to_matrix()
        self.to_dataframe()

    def _to_matrix(self):
        self.n_recipes = RECIPE_COLLECTION.find().count()
        ratings_dictionary = {}

        #recipes to numpy array and make dictionary of recipe_ids with index in matrix
        for r_idx, recipe in enumerate(RECIPE_COLLECTION.find(), start=0):
            recipe_id = recipe['recipe_id']
            self.recipe_idx[recipe_id] = r_idx

        for u_idx, user in enumerate(USER_COLLECTION.find().limit(self.n_users), start=0):
            user_id = user['user_id']
            self.user_idx[user_id] = u_idx
            for review in user['ratings']:
                if review.keys()[0] in self.recipe_idx:
                    r_idx = self.recipe_idx[review.keys()[0]]
                    ratings_dictionary[(u_idx, r_idx)] = review.values()[0]

        self.sparse_mat = sparse.dok_matrix((self.n_users, self.n_recipes))
        for idx, rating in ratings_dictionary.iteritems():
            self.sparse_mat[idx] = rating

        self.sparse_mat.transpose().tocsr()

    def to_dataframe(self):
        '''
        sends specified number of users to pandas dataframe.
        '''
        ary = np.zeros((1,3))
        for user in USER_COLLECTION.find().limit(self.n_users):
            user_id = user['user_id']
            for review in user['ratings']:
                tmp_ary = np.array([user_id, review.keys()[0], review.values()[0]])
                ary = np.vstack((ary, tmp_ary))
        self.df = pd.DataFrame(data=ary[1:,:], columns=['user_id', 'recipe_id', 'rating'], dtype='int64')

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



# if __name__ == '__main__':
#     d = DataLoader(1000)


    # def recipe_ratings_df():
    #     '''
    #     loads recipe id and ratings to Pandas DF from MongoDB
    #     '''
    #     n = RECIPE_COLLECTION.find().count()
    #     ary = np.zeros((n, 2))
    #     for i, recipe in enumerate(RECIPE_COLLECTION.find()):
    #         ary[i,0] = recipe['recipe_id']
    #         ary[i,1] = unidecode(recipe['rating'][0])
    #     df = pd.DataFrame(data=ary, columns=['recipe_id', 'rating'])
    #     df['recipe_id'] = df['recipe_id'].astype('int64')
    #     return df
