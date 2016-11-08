import numpy as np
import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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
    #send returned pandas df to sframe
    nlp_data = gl.SFrame(nlp_processor.tfidf.toarray())
    nlp_data.rename({'X1': 'recipe_id'})
    return nlp_data

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
    df = pd.read_pickle('../data_management/test_data.pkl')
    nlp_sf = get_nlp(df)
    sf = gl.SFrame(df)

    #train_test_split
    # train_set, test_set = sf.random_split(0.75, seed=42)
    train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)

    #control which functions to actually run here.
    baseline_model, train_rmse, test_rmse = baseline_model(train_set, test_set, item_data=nlp.sf)
    '''
    without nlp:

    with nlp:

    '''
    # trained_models = train_models(train_set, recommenders)

    models, models_rmse = kfolds(train_set, model_list=recommenders, model_names=recommender_names, item_data=nlp_sf)
    plot_error(models_rmse, save_as='test_nlp.jpg')
    plt.show()

    # fr = train_one(sf, recommender=gl.factorization_recommender)

    '''
    factorization_recommender 1.01312528315
    ranking_factorization_recommender 1.15281151156
    item_similarity_recommender 4.52668214059
    popularity_recommender 1.001186525
    '''

    # fr.save('model')
