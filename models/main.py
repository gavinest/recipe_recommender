import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import graphlab as gl
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management import DataLoader, ScoreMany
from nlp import NLPProcessor

'''
Central File - It has the ability to:
    - Load Graphlab ready NLP SFrame with help of custom NLP Classes
    - Plot score of a single trained model
    - Plot scores of many trained models
'''


def get_clusters(tfidf, features):
    '''
    class sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    '''

    kmeans = KMeans(n_clusters=50, init='k-means++', random_state=42, n_jobs=-1)
    kmeans.fit(tfidf)

    # 3. Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))
    return kmeans

def for_graphlab(nlp):
    features = nlp.vectorizer.get_feature_names()
    kmeans = get_clusters(nlp.tfidf, features)
    predictions = kmeans.predict(nlp.tfidf)
    predictions = pd.DataFrame(data=predictions, columns=['cluster'])
    df = nlp.taxonomy_to_df()
    df = pd.concat([df, predictions], axis=1)
    sf = gl.SFrame(df)
    return sf

def get_item_data(nlp=True, data_elimination=True):
    if nlp:
        nlp = NLPProcessor(vectorizer=TfidfVectorizer, kwargs={'analyzer':list, 'lowercase':False, 'max_df':0.8})
        nlp.fit_vectorizor()
        sf = for_graphlab(nlp)

    if data_elimination:
        pass
    return sf


if __name__ == '__main__':
    np.random.seed(seed=42)
    item_data_sf = get_item_data()

    # 'load data train test model'
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    sf = gl.SFrame(df)
    # count = df.groupby('user_id')['rating'].count()
    # df['rating_count'] = df['user_id'].apply(lambda x: count[x])
    # dfs = [df[df['rating_count'] > i] for i in range(4)]
    # sfs = [gl.SFrame(df) for df in dfs]
    # #
    # 'decide train test split type'
    train_set, test_set = sf.random_split(fraction=0.75, seed=42)
    # sets = [sf.random_split(fraction=0.75, seed=42) for sf in sfs]
    # # # train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)
    # #
    # # 'create or load models'
    # # #baseline model
    baseline = gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating')
    nlp_model = gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=item_data_sf)
    models = [baseline, nlp_model]
    colors = ['r', 'b']
    names = ['baseline', 'nlp']
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
    test = ScoreMany(sf=sf, test_set=test_set, recommenders=models, recommender_names=names, colors=colors)
    test.plot_score_all()
    # # test.score_precision_recall()
    # # # test._make_f1_lines
