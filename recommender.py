import numpy as np
import pandas as pd
from unidecode import unidecode
# import graphlab as gl
from pymongo import MongoClient
import matplotlib.pyplot as plt
plt.style.use('ggplot')



#pandas dataframe
# recipe_id | rating  | target
#-----------|---------|---------
#           |         | rating

# *rating will be calculated with separate weightings
# part actual rating, part how many deals it takes advantage of


#global variables
DB_NAME = 'allrecipes'
# COLL_NAME = 'recipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
DEALS_COLLECTION = DATABASE['deals']


def recipe_ratings_df():
    '''
    loads recipe id and ratings to Pandas DF from MongoDB
    '''
    n = RECIPE_COLLECTION.find().count()
    ary = np.zeros((n, 2))
    for i, recipe in enumerate(RECIPE_COLLECTION.find()):
        ary[i,0] = recipe['recipe_id']
        ary[i,1] = unidecode(recipe['rating'][0])
    df = pd.DataFrame(data=ary, columns=['recipe_id', 'rating'])
    df['recipe_id'] = df['recipe_id'].astype('int64')
    return df

def offer_tags():
    for offer in DEALS_COLLECTION.find():
        print offer['id'], offer['tags']



if __name__ == '__main__':
    # r = recipe_ratings_df()
    offer_tags()

    # plt.hist(df['rating'], bins=100)
    # plt.show()
