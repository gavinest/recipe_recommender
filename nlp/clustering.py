import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from nlp.clean_recipes import NLPProcessor


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
    return df


if __name__ == '__main__':
    nlp = NLPProcessor(vectorizer=TfidfVectorizer, kwargs={'analyzer':list, 'lowercase':False, 'max_df':0.8})
    nlp.fit_vectorizor()
    # features = nlp.vectorizer.get_feature_names()
    # with open("nlp_vectorizer.pkl") as f:
    #     nlp = pickle.load(f)
    # k = get_clusters(nlp.tfidf, features)
    df = for_graphlab(nlp)
