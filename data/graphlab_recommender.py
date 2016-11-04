import numpy as np
import pandas as pd
import graphlab as gl
from load_data import DataLoader

def gl_test(df):
    sf = gl.SFrame(df)

    # (train_set, test_set) = sf.random_split(0.8)
    model = gl.popularity_recommender.create(sf, user_id='user_id', item_id='recipe_id', target='rating')
    # prediction = model.predict(sf)
    return model

def recommend(model, user_id, n_recs=5):
    return model.recommend(gl.SFrame(user_df), k=n_recs)


if __name__ == '__main__':
    data = DataLoader(10)
    data.to_dataframe()
    df = data.df

    popularity_model = gl_test(df)
    # recs = recommend(popularity_model, user_id="827351")
    rec = popularity_model.recommend([827351], k=5)[0]['recipe_id']

    recipe_card = data.pull_recipe_data([str(rec)])



    # user_df = data.get_one_user("827351")

    # '46530'
