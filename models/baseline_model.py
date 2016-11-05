import numpy as np
import pandas as pd
import graphlab as gl
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader
from nlp.clean_recipes import NLPProcessor

'''for baseline model redo with gl.recommender.create()'''

def train_score_model(df, item_data=None):
    sf = gl.SFrame(df)

    #train_test_split
    (train_set, test_set) = sf.random_split(0.75, seed=42)

    #make_model
    model = gl.popularity_recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)

    #score
    rmse = gl.evaluation.rmse(targets=test_set['rating'], predictions=model.predict(test_set))
    return model, rmse


if __name__ == '__main__':
    #load users, recipe, and rating data
    data_loader = DataLoader(1000)
    data_loader.to_dataframe()
    df = data_loader.df
    #
    # #load nlp data
    # recipe_ids = df['recipe_id'].values
    # nlp_processor = NLPProcessor()
    # nlp_processor.make_tfidf(recipe_ids)
    # nlp_data = gl.SFrame(nlp_processor.tfidf.toarray())
    # nlp_data.rename({'X1': 'recipe_id'})
    #
    # model, rmse = train_score_model(df, item_data=nlp_data)

    # model.save('model')


    # #get results for bringing into web_app
    # test_user = 827351
    # recommendations = popularity_model.recommend([test_user], k=5)
    # predicted_ratings = [rec['score'] for rec in recommendations]
    #
    # #get recommended recipe cards from DataLoader
    # recommended_recipes = [rec['recipe_id'] for rec in recommendations]
    # recipe_cards = data_loader.pull_recipe_data(recommended_recipes)
