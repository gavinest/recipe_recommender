import requests
import unirest


# These code snippets use an open-source library.
# r = requests.get("https://spoonacular-recipe-food-nutrition-v1.p.mashape.com/food/products/classify",
#   # headers={
#   #   "X-Mashape-Key": "zUvf2kJg7vmshqUBQSgFjY0zeXP8p1pFtCgjsn70XpmzjlYZGI",
#   #   "Content-Type": "application/json",
#   #   "Accept": "application/json"
#   # },
#   params=("{\"title\":\"Kroger Vitamin A & D Reduced Fat 2% Milk\",\"upc\":\"\",\"plu_code\":\"\"}")
# )
response = unirest.post("https://spoonacular-recipe-food-nutrition-v1.p.mashape.com/food/products/classify",
  headers={
    "X-Mashape-Key": "zUvf2kJg7vmshqUBQSgFjY0zeXP8p1pFtCgjsn70XpmzjlYZGI",
    "Content-Type": "application/json",
    "Accept": "application/json"
  },
  params=("{\"title\":\"Kroger Vitamin A & D Reduced Fat 2% Milk\",\"upc\":\"\",\"plu_code\":\"\"}")
)
