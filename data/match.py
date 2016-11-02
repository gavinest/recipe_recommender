from pymongo import MongoClient

#global variables
DB_NAME = 'allrecipes'
# COLL_NAME = 'deals'
CLIENT = MongoClient()
DATABASE = CLIENT[DB_NAME]
# COLLECTION = DATABASE[COLL_NAME]


if __name__ == '__main__':
    main()
