from lxml import html

import unicodecsv as csv

import requests

from time import sleep

import re

import argparse

import json





def parse(url):

    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chrome/70.0.3538.77 Safari/537.36'}

    success = False

    

    for _ in range(10):

        response = requests.get(url, verify=False, headers=headers)

        if response.status_code == 200:

            success = True

            break

        else:

            print("Response received: %s. Retrying : %s"%(response.status_code, url))

            success = False

    

    if success == False:

        print("Failed to process the URL: ", url)

    

    parser = html.fromstring(response.text)

    listing = parser.xpath("//li[@class='regular-search-result']")

    raw_json = parser.xpath("//script[contains(@data-hypernova-key,'yelp_main__SearchApp')]//text()")

    scraped_datas = []

    

    # Case 1: Getting data from new UI

    if raw_json:

        print('Grabbing data from new UI')

        cleaned_json = raw_json[0].replace('<!--', '').replace('-->', '').strip()

        json_loaded = json.loads(cleaned_json)

        search_results = json_loaded['searchPageProps']['searchResultsProps']['searchResults']

        

        for results in search_results:

            # Ad pages doesn't have this key.  

            result = results.get('searchResultBusiness')

            if result:

                is_ad = result.get('isAd')

                price_range = result.get('priceRange')

                position = result.get('ranking')

                name = result.get('name')

                ratings = result.get('rating')

                reviews = result.get('reviewCount')

                address = result.get('formattedAddress')

                neighborhood = result.get('neighborhoods')

                category_list = result.get('categories')

                full_address = address+' '+''.join(neighborhood)

                url = "https://www.yelp.com"+result.get('businessUrl')

                

                category = []

                for categories in category_list:

                    category.append(categories['title'])

                business_category = ','.join(category)



                # Filtering out ads

                if not(is_ad):

                    data = {

                        'business_name': name,

                        'rank': position,

                        'review_count': reviews,

                        'categories': business_category,

                        'rating': ratings,

                        'address': full_address,

                        'price_range': price_range,

                        'url': url

                    }

                    scraped_datas.append(data)

        return scraped_datas
