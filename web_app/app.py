from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import graphlab as gl
from load_data import DataLoader

app = Flask(__name__)

@app.route('/')
@app.route('/about')
def hello_world():
    return render_template('summary.html')

@app.route('/contact')
def contact_me():
    return render_template('contact.html')

@app.route('/recommend')
def recommender():
    recipe_card = get_recipe_cards(df)[0]

    return render_template('recommender.html',
                            recipe_name=recipe_card['name'],
                            directions=recipe_card['directions'])

def get_recipe_cards(df):
    sf = gl.SFrame(df)
    model = gl.popularity_recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating')
    rec = model.recommend([827351], k=5)[0]['recipe_id']
    return data.pull_recipe_data([str(rec)])

if __name__ == '__main__':
    data = DataLoader(10)
    data.to_dataframe()
    df = data.df

    app.run(host='0.0.0.0', port=8080, debug=True)



    # time = [33, 63, 93, 123, 180, 240, 300, 470]
    # recipes = [750, 1110, 1434, 1945, 2400, 4340, 6123, 11694]
    # users = [39255, 52880, 65034, 81515, 94641, 131649, 159419, 218446]
