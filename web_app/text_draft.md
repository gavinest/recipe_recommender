## Text Drafts for Website Description and github README

# Ideas


## Description of Project
1. Idea

2. Problem it solves

3. Business Applications

### Data Wrangling
1. Scraping

2. Databasing

3. EDA


### Modeling

Goal of model: minimize RMSE or simply make a recommendation the user will like.

1. Graphlab (GL)
  a. Base model  
  GL's recommender package automatically selects a recommender model based on the input data. I allowed this functionality for GL to 'recommend' me a baseline recommender. GL created a RankingFactorizationRecommender from my data. I used as a baseline with the GL default hyper-parameters from which to compare model improvements.

2. Training the model

See Train-Test-Split_Comparison.jpg

Standard Train Test Split

```python
train_set, test_set = sf.random_split(0.75, seed=42)
```

Split for recommenders

```python
train_set, test_set = graphlab.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)
```

3. SCORING

http://aimotion.blogspot.com/2011/05/evaluating-recommender-systems.html



2. Feature engineering

  a. NLP
    - explain how it worked
  b. Reduce users under a certain number of reviews
    - show in graph how i arrived at the number
    - also hard to score your recommendation when someone has only reviewed one thing
    - _visuals_ graph mean rmse of model as drop users you have below a certain rating
  c. Additional Recipe Info
    - Add total recipe reviews and recipe average review to

Lastly, here are a couple of common data issues that can affect the performance of a recommender. First, if the observation data is very sparse, i.e., contains only one or two observations for a large number of users, then none of the models will perform much better than the simple baselines available via the popularity_recommender.

3. Model Improvement

### WebApp and making recommendations


### Notes
