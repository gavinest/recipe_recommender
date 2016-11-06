from flask import Flask, request, render_template, url_for, redirect
import numpy as np
import pandas as pd
import graphlab as gl
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


@app.route('/')
@app.route('/about')
def hello_world():
    return render_template('summary.html')

@app.route('/contact')
def contact_me():
    return render_template('contact.html')

@app.route('/recommender/<user_id>')
def recommender(user_id):
    recommendations = model.recommend([user_id], k=5)['recipe_id']
    recipe_cards =[]
    for recipe in recommendations:
        recipe_cards.append(RECIPE_COLLECTION.find_one({'recipe_id': str(recipe)}))
    return render_template('recommender.html', cards=recipe_cards)

@app.route('/login', methods=["GET", "POST"])
@app.route('/recommend')
@app.route('/recommender')
#827351
def login():
    error = ''
    if request.method == "POST":
        user_id = request.form['user_id']
        if USER_COLLECTION.find({'user_id': user_id}).count() != 0:
            return redirect(url_for('recommender', user_id=user_id))
        else:
            error = 'Invalid User ID. Do you want to register?'
    return render_template('login.html', error=error)

if __name__ == '__main__':
    data = DataLoader(10)
    data.to_dataframe()
    print data.user_idx
    df = data.df
    model = gl.load_model('../models/model')

    app.run(host='0.0.0.0', port=8080, debug=True)
