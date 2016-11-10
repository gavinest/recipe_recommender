import numpy as np
import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management import DataLoader, RecScorer
from nlp.clean_recipes import NLPProcessor
from collections import defaultdict
from pymongo import MongoClient

#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

#load nlp data
def get_nlp(df):
    print 'Getting NLP...'
    nlp_processor = NLPProcessor()
    df = nlp_processor.recipe_text_to_df(df['recipe_id'].values, filename='nlp_vectorizer.pkl')
    #send returned pandas df to sframe
    df = df[0].apply(lambda x: x.flatten())
    nlp_sf = gl.SFrame(df)
    nlp_sf.rename({'X1': 'recipe_id'})
    return df, nlp_sf

#get avg rating data
def get_avg_rating(df):
    print 'Getting avg rating data'
    avg_ratings = []
    for recipe_id in df['recipe_id'].values:
        card = RECIPE_COLLECTION.find_one({'recipe_id' : recipe_id})
        avg_ratings.append([float(card['rating'][0]), int(card['num_reviews'][0])])
    avg_rating_sf = gl.SFrame(np.array(avg_ratings))
    avg_rating_sf.rename({'X1': 'recipe_id'})
    return avg_rating_sf
    # return avg_ratings

def baseline_model(train_set, test_set, item_data=None):
    model = gl.recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)
    train_rmse = model['training_rmse']
    test_rmse = gl.evaluation.rmse(targets=test_set['rating'], predictions=model.predict(test_set))
    return model, train_rmse, test_rmse

def train_models(train_set, recommenders, item_data=None):
    trained_models = []
    for recommender in recommenders:
        model = recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)
        trained_models.append(model)
    return trained_models

def kfolds(sf, model_list, model_names, item_data=None):
    '''
    Input: List of models to train
    '''
    rmse_dct = defaultdict(list)
    folds = gl.cross_validation.KFold(sf, num_folds=5)
    for train_set, test_set in folds:
            trained_models = train_models(train_set, model_list, item_data=item_data)
            eval_results = gl.recommender.util.compare_models(test_set, models=trained_models, model_names=recommender_names, metric='rmse', target='rating')
            for i, result in enumerate(eval_results):
                rmse_dct[model_names[i]].append(result['rmse_overall'])
    return trained_models, rmse_dct

def train_one(sf, recommender, item_data=None):
    model = gl.recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data)
    return model

def plot_error(rmse_dct, save_as=None):
    names = rmse_dct.keys()
    rmses = [np.mean(_) for _ in rmse_dct.values()]
    colors = list('brgk')

    fig, axes = plt.subplots(2,2, figsize=(8,8), sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.scatter(1, rmses[i], color=colors[i])
        ax.set_title(names[i])

        ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    # plt.title('RMSE Comparison of Recommenders')
    # plt.legend(loc='best')
    plt.ylim(0, 5.0)
    plt.yticks(np.arange(0.0, 5.0, 0.5))

    if save_as:
        plt.savefig(save_as)

if __name__ == '__main__':
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

    #load data
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    sf = gl.SFrame(df)

    #train_test_split
    train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)

    trained_models = train_models(train_set, recommenders)
    fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, model in enumerate(trained_models):
        score = RecScorer(recommender=model, test_set=test_set)
        score.rmse_of_top_percent()
        axes[i] = score.ax
    plt.show()

    #load additional features here
    # nlp_df, nlp_sf = get_nlp(df)
    # nlp_df.to_pickle('nlp_1Hdata.pkl')
    # nlp_df = pd.read_pickle('nlp_1Hdata.pkl')
    # nlp_sf = gl.SFrame(nlp_df)
    # nlp_sf.rename({'X1': 'recipe_id'})

    # models, models_rmse = kfolds(sf, model_list=recommenders, model_names=recommender_names, item_data=None)
    # plot_error(models_rmse, save_as='test_nlp.jpg')
    # plt.show()

    # fr = train_one(sf, recommender=gl.factorization_recommender)

    '''
    models

    1h dataset (no item data)
    factorization_recommender 0.950535996995
    ranking_factorization_recommender 1.55995115886
    item_similarity_recommender 4.45193130763
    popularity_recommender 0.959436737018

    1h dataset (nlp_data)
    factorization_recommender 0.950092272217
    ranking_factorization_recommender 1.5511279349
    item_similarity_recommender 4.45193130763
    popularity_recommender 0.959436737018


    1K dataset
    without additional item data
    factorization_recommender 0.89364053205
    ranking_factorization_recommender 0.947981843138
    item_similarity_recommender 4.53224970378
    popularity_recommender 0.898589811666

    with nlp data:
    factorization_recommender 0.893915631704
    ranking_factorization_recommender 0.948105300499
    item_similarity_recommender 4.53225004309
    popularity_recommender 0.898589811666

    with avg_ratings
    factorization_recommender 0.893621611773
    ranking_factorization_recommender 0.947868617735
    item_similarity_recommender 4.53224992839
    popularity_recommender 0.898589811666

    ----

    10K dataset
    models with nlp: test_data test
    factorization_recommender 1.01640391824
    ranking_factorization_recommender 1.15228619831
    item_similarity_recommender 4.52668347749
    popularity_recommender 1.001186525

    models without nlp: test_data test
    factorization_recommender 1.01343448907
    ranking_factorization_recommender 1.15311581939
    item_similarity_recommender 4.52668307159
    popularity_recommender 1.001186525

    models with avg_ratings item data: test_data
    factorization_recommender 1.01744743967
    ranking_factorization_recommender 1.15266287541
    item_similarity_recommender 4.52668406905
    popularity_recommender 1.001186525

    '''

    # fr.save('model')
