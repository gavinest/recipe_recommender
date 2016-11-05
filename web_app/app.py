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

@app.route('/recommend', methods=["GET","POST"])
# @app.route('/recommend/<user_id>')
# def recommender(user_id=827351): #827351
def recommender():
    if request.method == "POST":
        user_id = request.form['user_id']
        if USER_COLLECTION.find({'user_id': user_id}).count() != 0:
            rec = model.recommend([945781], k=5)[0]['recipe_id']
            print rec
            recipe_card = data.pull_recipe_data([str(rec)])[0]
            return render_template('recommender.html',
                        recipe_name=recipe_card['name'],
                        ingredients=recipe_card['ingredients'],
                        directions=recipe_card['directions'])
    else:
        rec = model.recommend([827351], k=5)[0]['recipe_id']
        recipe_card = data.pull_recipe_data([str(rec)])[0]
        return render_template('recommender.html',
                    recipe_name=recipe_card['name'],
                    ingredients=recipe_card['ingredients'],
                    directions=recipe_card['directions'])

    # @app.route('/login/', methods=["GET","POST"])
    # def login_page():
    #
    #     error = ''
    #     try:
    #
    #         if request.method == "POST":
    #
    #             attempted_username = request.form['username']
    #             attempted_password = request.form['password']
    #
    #             #flash(attempted_username)
    #             #flash(attempted_password)
    #
    #             if attempted_username == "admin" and attempted_password == "password":
    #                 return redirect(url_for('dashboard'))
    #
    #             else:
    #                 error = "Invalid credentials. Try Again."
    #
    #         return render_template("login.html", error = error)
    #
    #     except Exception as e:
    #         #flash(e)
    #         return render_template("login.html", error = error)
    #

if __name__ == '__main__':
    data = DataLoader(10)
    data.to_dataframe()
    print data.user_idx
    df = data.df
    model = gl.load_model('../models/model')

    app.run(host='0.0.0.0', port=8080, debug=True)
