from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer

#terminal command for duplicating mongodb for testing updating
# mongoexport -d db -c sourcecollection | mongoimport -d db -c targetcollection --drop
# i.e. mongoexport -d allrecipes -c recipes | mongoimport -d allrecipes -c test --drop

#check mongo for duplicates
#db.users.aggregate({'$group': {'_id': '$user_id', 'count': {'$sum': 1}}}, {'$match': {'_id' :{'$ne': null}, 'count': {'$gt': 1}}}, {'$project': {'user_id': '$_id', '_id':0}})

#db.recipes.aggregate({'$group': {'_id': '$recipe_id', 'count': {'$sum': 1}}}, {'$match': {'_id' :{'$ne': null}, 'count': {'$gt': 1}}}, {'$project': {'recipe_id': '$_id', '_id':0}})

#GLOBAL VARIABLES
DB_NAME = 'allrecipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
RECIPE_COLLECTION = DATABASE['recipes']
USER_COLLECTION = DATABASE['users']

#CLEAN RECIPES
def stop_words():
    recipe_stopwords = set(['pound', 'pounds', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'cup', 'cups', 'bunch', 'chopped', 'diced', 'ounce', 'ounces'])
    stop = set(stopwords.words('english'))
    return stop.union(recipe_stopwords)

def clean_ingredients(recipe):
    stop = stop_words()
    cleaned = []
    for line in recipe:
        line = [_ for _ in unidecode(line).lower().translate(None, punctuation).split() if not _.isdigit()]
        lemmers = [WordNetLemmatizer().lemmatize(word) for word in line if word not in stop]
        cleaned.extend([word for word in lemmers])
    return cleaned

# def update_recipes():
#     for recipe in COLLECTION.find():
#         COLLECTION.update_one({'_id': recipe['_id']},
#                 {'$set': {'tokens': clean_ingredients(recipe['ingredients'])}},
#                 upsert=False)
#         # print clean_ingredients(recipe['ingredients'])
#     print 'Updated!'

def test_one():
    test = RECIPE_COLLECTION.find_one()
    text = test['ingredients']
    text.extend(test['taxonomy'])
    text.append(' '.join(test['name'].split('-')))
    print text
    clean = clean_ingredients(text)

    return test, clean

if __name__ == '__main__':
    # t = clean_ingredients(test_list)
    # update_recipes()
    test, clean = test_one()
