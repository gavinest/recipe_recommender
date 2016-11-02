from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from unidecode import unidecode

#terminal command for duplicating mongodb for testing updating
# mongoexport -d db -c sourcecollection | mongoimport -d db -c targetcollection --drop
# i.e. mongoexport -d allrecipes -c recipes | mongoimport -d allrecipes -c test --drop

#global variables
DB_NAME = 'allrecipes'
COLL_NAME = 'recipes'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
COLLECTION = DATABASE[COLL_NAME]

def stop_words():
    recipe_stopwords = set(['pound', 'pounds', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'cup', 'cups', 'bunch', 'chopped', 'ounce'])
    stop = set(stopwords.words('english'))
    return stop.union(recipe_stopwords)

def clean_ingredients(recipe):
    stop = stop_words()
    cleaned = []
    for line in recipe:
        line = [_ for _ in unidecode(line).lower().translate(None, punctuation).split() if not _.isdigit()]
        lemmers = [WordNetLemmatizer().lemmatize(word) for word in line if word not in stop]
        cleaned.append([unidecode(word) for word in lemmers])
    return cleaned

def update_recipes():
    for recipe in COLLECTION.find():
        COLLECTION.update_one({'_id': recipe['_id']},
                {'$set': {'tokens': clean_ingredients(recipe['ingredients'])}},
                upsert=False)
        # print clean_ingredients(recipe['ingredients'])
    print 'Updated!'

if __name__ == '__main__':
    test_list = [
		"6 pounds roma (plum) tomatoes",
		"1/4 pound roma (plum) tomatoes, chopped",
		"2 tablespoons garlic powder",
		"1/4 cup lemon juice",
		"1 1/2 tablespoons salt",
		"1 tablespoon ground cayenne pepper",
		"1 1/2 teaspoons ground cumin",
		"1 red onion, chopped",
		"1 white onion, chopped",
		"1 yellow onion, chopped",
		"1 pound jalapeno peppers, chopped",
		"1/3 bunch fresh cilantro, chopped"
	     ]
    # t = clean_ingredients(test_list)
    update_recipes()
