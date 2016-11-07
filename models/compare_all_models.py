import numpy as np
import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
import sys
# from sklearn.pipelines import Pipeline
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader
from nlp.clean_recipes import NLPProcessor
from collections import defaultdict

#load nlp data
def get_nlp(df):
    print 'Getting NLP...'
    nlp_processor = NLPProcessor()
    nlp_processor.make_tfidf(df['recipe_id'].values)
    #send return pandas df to sframe
    nlp_data = gl.SFrame(nlp_processor.tfidf.toarray())
    nlp_data.rename({'X1': 'recipe_id'})
    return nlp_data

def baseline_model(train_set, test_set, item_data=None):
    model = gl.recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)
    train_rmse = model['training_rmse']
    test_rmse = gl.evaluation.rmse(targets=test_set['rating'], predictions=model.predict(test_set))
    return model, train_rmse, test_rmse



# def score_models(train_set, test_set, recommenders, recommender_names=None):
#     #train_test_split
#     trained_models = train_models(train_set, recommenders)
#     rmse = []
#     for model in trained_models:
#         rmse.append(gl.evaluation.rmse(targets=test_set['rating'], predictions=model.predict(test_set)))
#
#     if recommender_names:
#         print 'Model    |   Score'
#         for name, score in zip(recommender_names, rmse):
#             print '{0}   |   {1}'.format(name, score)

def train_models(train_set, recommenders):
    trained_models = []
    for recommender in recommenders:
        model = recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating')
        trained_models.append(model)
    return trained_models

def kfolds(sf, model_list, model_names):
    '''
    Input: List of models to train
    '''
    rmse_dct = defaultdict(list)


    folds = gl.cross_validation.KFold(sf, num_folds=5)
    for train_set, test_set in folds:
            trained_models = train_models(train_set, model_list)
            eval_results = gl.recommender.util.compare_models(test_set, models=trained_models, model_names=recommender_names, metric='rmse', target='rating')
            for i, result in enumerate(eval_results):
                rmse_dct[model_names[i]].append(result['rmse_overall'])
    return rmse_dct

if __name__ == '__main__':
    '''
    graphlab recommender models:
        ['item_similarity_recommender',
        'item_content_recommender',
        'factorization_recommender',
        'ranking_factorization_recommender',
        'popularity_recommender']
    Models must already be trained before entering into the gl.recommender.util.compare_models()

    graphlab.recommender.util.compare_models(dataset, models, model_names=None, user_sample=1.0, metric='auto', target=None, exclude_known_for_precision_recall=True, make_plot=False, verbose=True, **kwargs)
    '''
    #list o' recommenders
    recommenders = [
            gl.item_similarity_recommender,
            gl.factorization_recommender,
            gl.ranking_factorization_recommender,
            gl.popularity_recommender,
            ]

    recommender_names = [
                        'item_similarity_recommender',
                        'factorization_recommender',
                        'ranking_factorization_recommender',
                        'popularity_recommender'
                        ]

    sf = gl.SFrame(pd.read_pickle('../eda/test_data.pkl'))

    #train_test_split
    # train_set, test_set = sf.random_split(0.75, seed=42)
    train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)

    #control which functions to actually run here.
    # baseline_model, train_rmse, test_rmse = baseline_model(train_set, test_set)
    # trained_models = train_models(train_set, recommenders)

    # s = gl.recommender.util.compare_models(test_set, models=trained_models, model_names=recommender_names, metric='rmse', target='rating')
    # print 'Model    |   Score'
    # for name, result in zip(recommender_names, s):
    #     print '{0}   |   {1}'.format(name, result['rsme_overall'])

    d = kfolds(sf, model_list=recommenders, model_names=recommender_names)

    '''
    rmse_overall according to GL
    item_similarity_recommender 3.7831458378
    factorization_recommender 0.0270346385972
    ranking_factorization_recommender 0.101505661109
    popularity_recommender 0.33221817048

    rmse_overall for baseline_model according to GL
    #baseline_model.evaluate_rmse(test_set, target='rating')
    RankingFactorizationRecommender
    'rmse_overall': 0.7197888635612298}

    rmse_according to my scoring function
    Model    |   Score
    item_similarity_recommender   |   4.36269213092
    factorization_recommender   |   1.03286681888
    ranking_factorization_recommender   |   1.81056636023
    popularity_recommender   |   1.06330192504
    '''

    # model.save('model')

    # #get results for bringing into web_app
    # test_user = 827351
    # recommendations = popularity_model.recommend([test_user], k=5)
    # predicted_ratings = [rec['score'] for rec in recommendations]
    #
    # #get recommended recipe cards from DataLoader
    # recommended_recipes = [rec['recipe_id'] for rec in recommendations]
    # recipe_cards = data_loader.pull_recipe_data(recommended_recipes)
