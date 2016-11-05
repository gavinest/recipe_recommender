import numpy as np
import pandas as pd
import graphlab as gl
import sys
from sklearn.pipelines import Pipeline
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader
from nlp.clean_recipes import NLPProcessor

def train_models(train_set, recs):
    models = []
    for rec in recs:
        rec.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)
        models.append(rec)

if __name__ == '__main__':
    '''
    graphlab recommender models:
        item_similarity_recommender
        item_content_recommender
        factorization_recommender
        ranking_factorization_recommender
        popularity_recommender

    Models must already be trained before entering into the gl.recommender.util.compare_models()

    graphlab.recommender.util.compare_models(dataset, models, model_names=None, user_sample=1.0, metric='auto', target=None, exclude_known_for_precision_recall=True, make_plot=False, verbose=True, **kwargs)
    '''
    #list o' recommenders
        recommenders = [gl.item_similarity_recommender(user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)),
                    ('item_content',
                    gl.item_content_recommender(user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)),
                    ('factorization',
                    gl.factorization_recommender(user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)),
                    ('ranking_factorization',
                    gl.ranking_factorization_recommender(user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)),
                    ('popularity',
                    gl.popularity_recommender(user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data))]

    #train_test_split
    (train_set, test_set) = sf.random_split(0.75, seed=42)

    models = train_models(train_set, recommenders)

    #scoring already trained models
    gl.recommender.util.compare_models()
