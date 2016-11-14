import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import graphlab as gl
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management import DataLoader, RecScorer, ScoreMany
from nlp import NLPProcessor

'''
Main file for scoring/ optimizing reommenders.
    Brings in custom classes to add more predictors to models if desired and score multiple models at once.

    It has the ability to:
        - Load Graphlab ready NLP SFrame with help of custom NLP Classes
        - Plot score of a single trained model
        - Plot scores of many trained models
'''

def get_item_data(nlp=True):
    '''
    Inputs: Bools. Specify which item data you would like added to the item data sf. Default is all that are available.
    Output: Graphlab SFrame
    '''
    if nlp:
        #create and fit tfidfvectorizer on all text from all recipes in database
        nlp = NLPProcessor(vectorizer=TfidfVectorizer, kwargs={'analyzer':list, 'lowercase':False, 'max_df':0.8})
        nlp.fit_vectorizor()

        #create clusters from tfidf
        kmeans = KMeans(n_clusters=50, init='k-means++', random_state=42, n_jobs=-1)
        kmeans.fit(nlp.tfidf)

        #predict groupings from recipes & send to sf
        predictions = kmeans.predict(nlp.tfidf)
        predictions = pd.DataFrame(data=predictions, columns=['cluster'])
        df = nlp.taxonomy_to_df()
        df = pd.concat([df, predictions], axis=1)
        sf = gl.SFrame(df)
    return sf

def format_for_scoring_data_elimination(recommenders, recommender):
    '''
    Inputs:
        recommenders: Type=List of trained recommender objects

    Ouput:
        Type=List ready to unpack in ScoreMany Object for plotting.

    sf, test_set, recommenders, recommender_names, colors
    '''

def score_vs_baseline():
    pass



if __name__ == '__main__':
    np.random.seed(seed=42)
    item_data_sf = get_item_data()

    # 'load data train test model'
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    sf = gl.SFrame(df)

    df1 = df.copy()
    count = df1.groupby('user_id')['rating'].count()
    df1['rating_count'] = df['user_id'].apply(lambda x: count[x])
    df1 = df1[df1['rating_count'] > 1]
    sf1 = gl.SFrame(df1)
    # dfs = [df[df['rating_count'] > i] for i in range(4)]
    # sfs = [gl.SFrame(df) for df in dfs]
    # #
    # 'decide train test split type'
    train_set, test_set = sf.random_split(fraction=0.75, seed=42)
    train_set1, test_set1 = sf1.random_split(fraction=0.75, seed=42)
    # sets = [sf.random_split(fraction=0.75, seed=42) for sf in sfs]
    # # # train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)
    # #
    # # 'create or load models'
    baseline = gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating')
    nlp_model = gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data_sf)
    nlp_data_elimination = gl.factorization_recommender.create(train_set1, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data_sf)
    models = [baseline, nlp_model, nlp_data_elimination]
    colors = ['r', 'b', 'g']
    names = ['baseline', 'nlp', 'nlp_with_data_elimination']
    # # # model.save('scorer_test_model')
    # #
    # # regularization_vals = [0.001, 0.0001, 0.00001, 0.000001]
    # # names = ['r={}'.format(r) for r in regularization_vals]
    # names = ['>= {}'.format(i) for i in range(1,5)]
    # c = plt.get_cmap('viridis')
    # n = len(names)
    # colors = [c(float(i)/n) for i in range(n)]
    # #
    # models = [gl.factorization_recommender.create(sets[i][0], user_id='user_id', item_id='recipe_id', target='rating', item_data=None, random_seed=42)for i in range(4)]
    # # # model = gl.load_model('scorer_test_model')
    test = ScoreMany(sfs=[sf, sf, sf1], test_sets=[test_set, test_set, test_set1], recommenders=models, recommender_names=names, colors=colors)
    test.plot_score_all()
    # # test.score_precision_recall()
    # # # test._make_f1_lines
    # t = RecScorer(recommender=nlp_model, sf=sf, test_set=test_set)
    # t.score_precision_recall(plot=True)
