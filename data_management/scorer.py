import numpy as np
import pandas as pd
import graphlab as gl

class RecScorer(object):
    '''
    Scores on top p percent of jokes recommended to each user. Looks at the actual ratings (in the test data) that the user gave those jokes. Your score is the average of those ratings.

    To score well, recomender only needs to identify which jokes a user is likely to rate most highly (so the absolute accuracy of your ratings is less important than the rank ordering).
    '''

    def __init__(self, recommender, test_set, p=1.0):
        '''
        Input: trained graphlab recommender object
                Graphlab SFrame of Test Data
        '''
        self.recommender = recommender
        self.n_users = recommender.num_users
        self.sf = test_set

    def rmse_of_top_percent(self):
        users = set(self.sf['user_id'])
        recs = self.recommender.recommend(list(users), k=100)
        return recs

    def match_ids(self, recs):
        for pair in recs['recipe_id']


if __name__ == '__main__':
    #load data train test model
    df = pd.read_pickle('data.pkl')
    sf = gl.SFrame(df)
    # train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)
    model = gl.factorization_recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=None)
    train_rmse = model['training_rmse']

    test = RecScorer(recommender=model, test_set=sf)
    t = test.rmse_of_top_percent()
