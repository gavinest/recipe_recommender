import numpy as np
import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('ggplot')
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader
from nlp.clean_recipes import NLPProcessor
from collections import defaultdict
from pymongo import MongoClient

#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

def plot_split_comparision(regular_split, split_by_user):
    '''
    inputs of the form: list-of-lists
    [[#reviews, train_rmse, test_rmse, n_users_reamingin], ...]
    '''
    test_line = mlines.Line2D([],[],color='r', label='Regular Split')
    train_line = mlines.Line2D([], [], color='b', label='Split By User')
    remaining_line = mlines.Line2D([], [], color='k', label='Fraction Remaining Users', alpha=0.5)
    dash_line = mlines.Line2D([],[], color='k', linestyle='--', label='*Denotes Training Set')

    fig, ax = plt.subplots(1, figsize=(8,8))

    data = zip(*regular_split)
    ax.plot(data[0], data[1], color='r', linestyle='--')
    ax.plot(data[0], data[2], color='r', linestyle='-')
    data = zip(*split_by_user)
    ax.plot(data[0], data[1], color='b', linestyle='--')
    ax.plot(data[0], data[2], color='b', linestyle='-')

    plt.ylabel('RMSE')
    plt.xlabel('# of Ratings per User is greater or equal to')
    plt.twinx(ax=ax)
    ax.plot(data[0], np.array(data[3]) / float(9999), color='k', alpha=0.5)
    plt.ylabel('Fraction of Original Users Remaining in Data Set')
    plt.title('Train-Test-Split Comparison')
    ax.legend(handles=[train_line, test_line, remaining_line, dash_line], loc='best')
    plt.savefig('Train-Test-Split_Comparison.jpg')
    plt.show()

def plot(df):
    df_users = df.groupby('user_id')['recipe_id']
    num_users = df_users.count().shape[0]
    print 'Number of Users:', num_users

    percent_users_remaining = []
    for i in xrange(0, 101):
        percent_users_remaining.append(df_users.count()[df_users.count() >= i].shape[0] / float(num_users))


    plt.plot(range(0, 101), percent_users_remaining)
    plt.title('Users with # Reviews or Greater')
    plt.xlabel('# of Reviews')
    plt.ylabel('Percent')
    plt.show()

def optimize_num_user_ratings(recommender, df, num_reviews=range(1,6)):
    rmses = []
    df_users = df.groupby('user_id')['recipe_id']
    for n in num_reviews:
        temp_df = df.copy()
        s = set(df_users.count()[df_users.count() >= n].index)
        temp_df['bool'] = temp_df['user_id'].apply(lambda x: x in s)
        temp_df[temp_df['bool'] == True]
        temp_df.drop(['bool'], axis=1, inplace=True)

        sf = gl.SFrame(temp_df)
        # train_set, test_set = sf.random_split(0.75, seed=42)
        train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)
        model = recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=None)
        train_rmse = model['training_rmse']
        test_rmse = gl.evaluation.rmse(targets=test_set['rating'], predictions=model.predict(test_set))

        rmses.append([n, train_rmse, test_rmse, len(s)])

    return rmses


if __name__ == '__main__':
    recommender = gl.factorization_recommender

    #load data
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    # df['rating'] = df['rating'].apply(lambda x: x**2)

    t = optimize_num_user_ratings(recommender, df, num_reviews=range(1,21))

    # regular_split = [[1, 0.24756219605916, 1.2812281896779578, 9999],
    # [2, 0.24396506562872577, 1.2919684140970615, 6539],
    # [3, 0.2456772661176323, 1.280217958169153, 5282],
    # [4, 0.24886309460966258, 1.2896848422016476, 4519],
    # [5, 0.2465071158587426, 1.2850042837499056, 3970],
    # [6, 0.24598692512622802, 1.285122490645728, 3548],
    # [7, 0.24550782044752326, 1.2978213537056298, 3190],
    # [8, 0.24693389954533243, 1.2763835962219936, 2895],
    # [9, 0.2469576193642692, 1.280561812570695, 2635],
    # [10, 0.24416307934024578, 1.282506642683546, 2436],
    # [11, 0.24291880542558184, 1.2786735958231497, 2259],
    # [12, 0.24677413170467127, 1.2944176098460718, 2094],
    # [13, 0.2454470018927928, 1.28547122878164, 1941],
    # [14, 0.2463482201053465, 1.2860904094230663, 1832],
    # [15, 0.24631757905664298, 1.2924017401092736, 1695],
    # [16, 0.24763016380128142, 1.2720425358474545, 1594],
    # [17, 0.2487463986264541, 1.2963493955821084, 1493],
    # [18, 0.24641071859198532, 1.2937212143499095, 1390],
    # [19, 0.249170947503105, 1.2888994583574553, 1319],
    # [20, 0.2463897120732691, 1.280820179605547, 1240]]
    #
    # split_by_user = [[1, 0.3888515858791448, 1.1587413359698964, 9999],
    # [2, 0.3871162588302974, 1.190586291913686, 6539],
    # [3, 0.3860211406094465, 1.1214529697877547, 5282],
    # [4, 0.38694060087332754, 1.1362120294998048, 4519],
    # [5, 0.3884719101211021, 1.1409012556785347, 3970],
    # [6, 0.3864935053728963, 1.1610899842723548, 3548],
    # [7, 0.38627642632627807, 1.1709433642015756, 3190],
    # [8, 0.3919026992211776, 1.1582203500653714, 2895],
    # [9, 0.3846179588710105, 1.1176034795858754, 2635],
    # [10, 0.384968520413604, 1.1230133963217572, 2436],
    # [11, 0.38480154129496225, 1.1439855757163433, 2259],
    # [12, 0.3388207847174976, 1.3325450428446144, 2094],
    # [13, 0.3864440422331071, 1.1523923213744052, 1941],
    # [14, 0.3850278827595873, 1.1388107678257164, 1832],
    # [15, 0.3860732719396107, 1.1492947361498353, 1695],
    # [16, 0.38629785698473507, 1.1607456938948177, 1594],
    # [17, 0.387744239806593, 1.1402237790809875, 1493],
    # [18, 0.38535178422160926, 1.1318140416241094, 1390],
    # [19, 0.38727471653082024, 1.1598174465469326, 1319],
    # [20, 0.3838702710895922, 1.1507788224106603, 1240]]

    # fr.save('model')
