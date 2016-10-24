from pymongo import MongoClient
from unidecode import unidecode
from string import punctuation
# import nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_ingredients():
    '''
    output: list of lists
        [ingredient , quantity(float) , unit , optional (1 or 0)]
    '''
    #access database
    db_client = MongoClient()
    db = db_client['allrecipes']
    collection = db['recipes']

    #clean ingredients
    documents = []
    for doc in collection.find():
        doc_txt = []
        for line in doc['ingredients']:
            quantity, words = [], []
            line_lst = unidecode(line).lower().translate(None, punctuation).split(' ')
            for item in line_lst:
                if item.isdigit():
                    quantity.append(item)
                else:
                    words.append(item)
            doc_txt.append(' '.join(words))
        documents.append(doc_txt)
    return documents



if __name__ == '__main__':
    docs = parse_ingredients()
