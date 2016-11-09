from pymongo import MongoClient
import pandas as pd
import time
import graphlab as gl
import numpy as np

#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

def get_avg_rating(recipe_ids):
    print 'Getting avg rating data'
    avg_ratings = []
    for recipe_id in recipe_ids:
        card = RECIPE_COLLECTION.find_one({'recipe_id' : recipe_id})
        avg_ratings.append([float(card['rating'][0]), int(card['num_reviews'][0])])
    avg_rating_sf = gl.SFrame(np.array(avg_ratings))
    avg_rating_sf.rename({'X1': 'recipe_id'})
    return avg_rating_sf
    # return avg_ratings

if __name__ == '__main__':
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
    s = time.time()
    cursor = RECIPE_COLLECTION.find({'recipe_id': {'$in': test_ids}}, {'rating': 1, 'num_reviews': 1})
    df = pd.DataFrame([c for c in cursor])
    df.drop(['_id'], axis=1, inplace=True)
    df['rating'] = df['rating'].apply(lambda x: x[0]).astype('float64')
    df['num_reviews'] = df['num_reviews'].apply(lambda x: x[0]).astype('int64')
    sf = gl.SFrame(df)
    print time.time() - s

    s1 = time.time()
    sf1 = get_avg_rating(test_ids)
    print time.time() - s1
