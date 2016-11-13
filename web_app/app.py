from flask import Flask, request, render_template, url_for, redirect
from wtforms import Form
import numpy as np
import pandas as pd
import graphlab as gl
from unidecode import unidecode
from pymongo import MongoClient
import sys
sys.path.append('/Users/Gavin/ds/recipe_recommender')
from data_management.load_data import DataLoader

app = Flask(__name__)
#global variables
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']


@app.route('/login', methods=["GET", "POST"])
@app.route('/recommend', methods=["GET", "POST"])
@app.route('/recommender', methods=["GET", "POST"])
@app.route('/', methods=["GET", "POST"])
#827351
def login():
    error1 = error2 = None
    if request.method == "POST":
        if 'user_id' in request.form:
            user = request.form['user_id']
            if USER_COLLECTION.find({'user_id': user}).count() != 0:
                return redirect(url_for('recommender', user_id=user))
            else:
                error1 = True
        elif 'register' in request.form:
            new_user = request.form['register']
            if USER_COLLECTION.find({'user_id': new_user}).count()!= 0:
                error2 = 'Invalid username. User {} already exists.'.format(new_user)
            else:
                USER_COLLECTION.insert_one({'user_id': new_user})
                error2 = 'Success. Welcome User {}.'.format(new_user)
    return render_template('login.html', intro_header='The Recipe Recommender',  tag_line='Delicious Food is Just a Click Away', error1=error1, error2=error2, random_users=get_random_ids())


# @app.route('/about')
# def hello_world():
#     return render_template('summary.html', intro_header='Recipe Recommender', tag_line='Make Good Food. Save Money.')

@app.route('/contact')
def contact_me():
    return render_template('contact.html', intro_header='Gavin Estenssoro', tag_line='Data Scientist | Engineer')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'POST':
        username = request.form['name']
    return render_template('register.html')

@app.route('/recipe/<recipe_id>')
def find_recipe(recipe_id):
    recipe_card = RECIPE_COLLECTION.find_one({'recipe_id': str(recipe_id)})
    return render_template('recipe_card.html', card=recipe_card)

@app.route('/recommender/<user_id>')
def recommender(user_id, page=1):
    #make page specific dict
    page_dct = {
                1: [0,9],
                2: [9,18],
                3: [18,27],
                4: [27,36],
                5: [36,45]
                }
    idx = page_dct[int(page)]

    recommendations = model.recommend([user_id], k=45)['recipe_id']
    recipe_cards =[]
    for recipe in recommendations:
        recipe_cards.append(RECIPE_COLLECTION.find_one({'recipe_id': str(recipe)}))
    return render_template('recommender.html', user_id=int(user_id), cards=recipe_cards[idx[0]:idx[1]], avg_ratings=[round(float(card['rating'][0]),2) for card in recipe_cards][idx[0]:idx[1]])

@app.route('/recommender/<user_id>/<page>')
def new_page(user_id, page=1):
    return redirect(url_for('recommender', user_id=user_id, page=page))

def get_random_ids():
    s = list(set(df['user_id'].values))
    users = np.random.choice(np.array(s), size=5, replace=False)
    return [int(unidecode(user)) for user in users]

if __name__ == '__main__':
    # data = DataLoader(10)
    # data.to_dataframe()
    # print data.user_idx
    # df = data.df
    df = pd.read_pickle('../data_management/pkls/data.pkl')
    model = gl.load_model('../models/model')

    app.run(host='0.0.0.0', port=5000, debug=True)
