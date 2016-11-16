import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import graphlab as gl
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management import DataLoader, RecScorer
from nlp import NLPProcessor

'''
Main file for scoring/ optimizing reommenders.
    Brings in custom classes to add more predictors to models if desired and score multiple models at once.

    It has the ability to:
        - Load Graphlab ready NLP SFrame with help of custom NLP Classes
        - Plot score of a single trained model
        - Plot scores of many trained models
'''

def train_many_models(sf, recommender_types, kwargs_lst=[{'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':None}]):
    '''
    Only works for models being trained/ tested on same data set.
    '''
    trained_models = []
    for i, recommender in enumerate(recommender_types):
        model = recommender.create(train_set, **kwargs_lst[i])
        trained_models.append(model)
    return trained_models

def get_data_eliminations_set(df, i=1):
    '''
    Input:
        DataFrame. Original Pandas dataframe.
        Int. Eliminate users with number of ratings less than int.
    Output:
        DataFrame
    '''
    count = df.groupby('user_id')['rating'].count()
    df['rating_count'] = df['user_id'].apply(lambda x: count[x])
    df = df[df['rating_count'] > i]
    df.drop('rating_count', axis=1, inplace=True)
    # dfs = [df[df['rating_count'] > i] for i in range(4)]
    # sfs = [gl.SFrame(df) for df in dfs]
    return df

def get_item_data(nlp=True, taxonomy=True, avg_ratings=True):
    '''
    Inputs: Bools. Specify which item data you would like added to the item data sf. Default is all that are available.
    Output: Graphlab SFrame
    '''
    dfs = []
    if nlp:
        #create and fit tfidfvectorizer on all text from all recipes in database
        nlp = NLPProcessor(vectorizer=TfidfVectorizer, kwargs={'analyzer':list, 'lowercase':False, 'max_df':0.8})
        nlp.fit_vectorizor()

        #create clusters from tfidf
        kmeans = KMeans(n_clusters=50, init='k-means++', random_state=42, n_jobs=-1)
        kmeans.fit(nlp.tfidf)

        #predict groupings from recipes & send to sf
        predictions = kmeans.predict(nlp.tfidf)
        #add axes to stack together
        predictions = predictions[:, np.newaxis]
        recipes = np.array(nlp.recipes_tfidf)[:, np.newaxis]
        pred_df = pd.DataFrame(data=np.hstack((recipes, predictions)), columns=['recipe_id','cluster'])
        pred_df['recipe_id'] = pred_df['recipe_id'].astype('int64')
        dfs.append(pred_df)

    d = DataLoader()
    if taxonomy:
        tax_df = d.taxonomy_to_df()
        tax_df['recipe_id'] = tax_df['recipe_id'].astype('int64')
        dfs.append(tax_df)

    if avg_ratings:
        rating_df = d.avg_rating_df()
        rating_df['recipe_id'] = rating_df['recipe_id'].astype('int64')
        dfs.append(rating_df)

    if len(dfs) > 1:
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = final_df.merge(df, on='recipe_id')
        return final_df
    else:
        return df[0]

def score_many_models(sfs, test_sets, recommenders, recommender_names, colors, new_item_data=None):
    '''
    Inputs:
        recommenders: Type=List of trained recommender objects
    Ouput:
        Type=List ready to unpack in ScoreMany Object for plotting.

    sf, test_set, recommenders, recommender_names, colors
    '''

    fig, axes = plt.subplots(1,2, figsize=(14,6))

    scorer_objs = []
    for i, recommender in enumerate(recommenders):
        R = RecScorer(sf=sfs[i], test_set=test_sets[i], new_item_data=new_item_data[i], recommender=recommender, name=recommender_names[i], color=colors[i], axes=axes, i=i+1)
        R.score_precision_recall()
        scorer_objs.append(R)
    # scorer_objs[0].axes[1].plot(range(1,5), [obj.test_rmse for obj in scorer_objs], color='r')
    scorer_objs[0]._make_f1_lines()

def bar_rmse(sfs, test_sets, new_item_data, recommenders, recommender_names, colors):
    rmses = []
    for i, recommender in enumerate(recommenders):
        rmses.append(gl.evaluation.rmse(targets=test_sets[i]['rating'], predictions=recommender.predict(test_sets[i], new_item_data=new_item_data[i])))

    score_dct = dict(zip(rmses, recommender_names))
    sorted_rmses = score_dct.keys()
    sorted_rmses = sorted(sorted_rmses, reverse=True)
    print sorted_rmses
    fig, ax = plt.subplots(1, figsize=(8,8))
    for i, rmse in enumerate(sorted_rmses):
        ax.bar(i, rmse, align='center', color=colors[i], alpha=0.8)
        ax.annotate('{0:.2f}'.format(rmse), xy=(i-0.05, rmse+0.005), textcoords='data')
        ax.set_ylabel('RMSE')
        ax.set_title('Recommender Test RMSE')

    plt.xticks(range(len(recommenders)), [score_dct[rmse] for rmse in sorted_rmses])

if __name__ == '__main__':
    np.random.seed(seed=42)
    #
    # '''get extra item data'''
    item_data = get_item_data(nlp=True, taxonomy=True, avg_ratings=True)

    '''recipe_id cluster  taxonomy_id  avg_rating  num_reviews'''

    #
    # '''load data train test model'''
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    # df['recipe_id'] = df['recipe_id'].astype('int64')
    # df = item_data.merge(df, on='recipe_id')
    # sf = gl.SFrame(df)

    '''get data elimination sets if applicable'''
    df = get_data_eliminations_set(df.copy(), i=1)
    df['recipe_id'] = df['recipe_id'].astype('int64')
    df = item_data.merge(df, on='recipe_id')
    sf = gl.SFrame(df)

    #decide on train test split type
    train_set, test_set = sf.random_split(fraction=0.75, seed=42)
    # train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.50, random_seed=42)

    #create item data from split
    nlp_item_data_sf = train_set['recipe_id', 'cluster']
    tax_item_data_sf = train_set['recipe_id', 'taxonomy_id']
    avg_rating_item_data_sf = train_set['recipe_id', 'avg_rating', 'num_reviews']
    all_item_data_sf = train_set['recipe_id', 'cluster', 'taxonomy_id', 'avg_rating', 'num_reviews']

    train_set = train_set['recipe_id', 'user_id', 'rating']

    #create new_item_data for model predictions
    test_nlp_item_data_sf = test_set['recipe_id', 'cluster']
    test_tax_item_data_sf = test_set['recipe_id', 'taxonomy_id']
    test_avg_rating_item_data_sf = test_set['recipe_id', 'avg_rating', 'num_reviews']
    test_all_item_data_sf = test_set['recipe_id', 'cluster', 'taxonomy_id', 'avg_rating', 'num_reviews']
    test_item_data = [None, test_nlp_item_data_sf, test_tax_item_data_sf, test_avg_rating_item_data_sf, test_all_item_data_sf]

    # '''create or load models'''
    models = [gl.factorization_recommender] * 5
    models_kwargs = [{'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':None},
                     {'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':nlp_item_data_sf},
                     {'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':tax_item_data_sf},
                     {'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':avg_rating_item_data_sf},
                    {'user_id':'user_id', 'item_id':'recipe_id', 'target':'rating', 'item_data':all_item_data_sf}]

    # models_data_elim = [gl.factorization_recommender] * 2
    #
    trained_models = train_many_models(train_set, recommender_types=models, kwargs_lst=models_kwargs)
    # data_elim_models, data_elim_test_sets = train_many_models(sf1, recommender_types=models, kwargs_lst=models_kwargs)
    #
    # '''format for scoring'''
    colors = ['r', 'b', 'k', 'g', 'm']
    names = ['baseline', 'nlp data', 'taxonomy data', 'avg rating data', 'all features']
    sfs = [sf['recipe_id', 'user_id', 'rating']] * 5
    test_sets = [test_set] * 5
    # # # regularization_vals = [0.001, 0.0001, 0.00001, 0.000001]
    # # # names = ['r={}'.format(r) for r in regularization_vals]
    # # names = ['>= {}'.format(i) for i in range(1,5)]
    # c = plt.get_cmap('viridis')
    # n = len(names)
    # colors = [c(float(i)/n) for i in range(n)]
    #
    # '''score/plot'''
    bar_rmse(sfs=sfs, test_sets=test_sets, new_item_data=test_item_data, recommenders=trained_models, recommender_names=names, colors=colors)
    test = score_many_models(sfs=sfs, test_sets=test_sets, new_item_data=test_item_data, recommenders=trained_models, recommender_names=names, colors=colors)

    '''decide on final model. save here for use in web_app'''
    # # # model.save('scorer_test_model')
    # #
    # models = [gl.factorization_recommender.create(sets[i][0], user_id='user_id', item_id='recipe_id', target='rating', item_data=None, random_seed=42)for i in range(4)]
    # # # model = gl.load_model('scorer_test_model')
