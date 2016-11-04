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
class NLPProcessor(object):
    def __init__(self, vectorizer=TfidfVectorizer, kwargs=None):
        self.documents = []
        self.vectorizer = vectorizer()

    def _stop_words(self):
        recipe_stopwords = set(['pound', 'pounds', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'cup', 'cups', 'bunch', 'chopped', 'diced', 'ounce', 'ounces'])
        stop = set(stopwords.words('english'))
        return stop.union(recipe_stopwords)

    def _tokenize(self, text):
        stop = self._stop_words()
        tokens = []
        for line in text:
            line = [_ for _ in unidecode(line).lower().translate(None, punctuation).split() if not _.isdigit()]
            lemmers = [WordNetLemmatizer().lemmatize(word) for word in line if word not in stop]
            tokens.extend([word for word in lemmers])
        return tokens

'''update database with recipe 'tokens' or this shit will take a real long time everytime it runs'''

    def get_tfidf(self, ids):
        for recipe_id in ids:
            recipe = RECIPE_COLLECTION.find_one({'recipe_id': recipe_id})
            print recipe
            if not recipe:
                text = recipe['ingredients']
                text.extend(recipe['taxonomy'])
                text.append(' '.join(recipe['name'].split('-')))
                self.documents.append(' '.join(self._tokenize(text)))
            else:
                self.documents.append(' ')

        self.tfidf = self.vectorizer.fit_transform(self.documents)

    # def test(self, num_recipes=1):
    #     documents = []
    #     for recipe in RECIPE_COLLECTION.find().limit(num_recipes):
    #         text = recipe['ingredients']
    #         text.extend(recipe['taxonomy'])
    #         text.append(' '.join(recipe['name'].split('-')))
    #         documents.append(' '.join(tokenize(text)))
    #     return documents

# def update_recipes():
#     for recipe in COLLECTION.find():
#         COLLECTION.update_one({'_id': recipe['_id']},
#                 {'$set': {'tokens': clean_ingredients(recipe['ingredients'])}},
#                 upsert=False)
#         # print clean_ingredients(recipe['ingredients'])
#     print 'Updated!'


if __name__ == '__main__':
    # t = clean_ingredients(test_list)
    # update_recipes()
    t = NLPProcessor()
