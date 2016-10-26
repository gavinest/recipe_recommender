from bs4 import BeautifulSoup
import urllib, urllib2
import requests

#ibotta.com uses ajax
#offers are stored in offers.jsob
# in google dev tools
# -> network -> XHR
#https://www.datascraping.co/doc/questions/28/how-do-i-crawl-an-infinite-scrolling-ajax-website

def scrape(url):
    response = urllib2.urlopen(url)
    # r = requests.get(url)
    # print r.status_code
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')

    return soup

if __name__ == '__main__':
    url = 'https://ibotta.com/rebates?category=grocery'
    soup = scrape(url)
