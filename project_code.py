import os
import re
import string

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer



df_yelp_business = pd.read_json('../input/yelp_academic_dataset_business.json', lines=True)
df_yelp_business.fillna('NA', inplace=True)
# we want to make sure we only work with restaurants -- nothing else
df_yelp_business = df_yelp_business[df_yelp_business['categories'].str.contains('Restaurants')]
print('Final Shape: ',df_yelp_business.shape)


#Now we bring the reviews and perform some preprocessing on those reviews..



df_yelp_review_iter = pd.read_json('../input/yelp_academic_dataset_review.json', chunksize=100000, lines=True)

#Note here we choose 5 chunks, but we could have chosen any number (larger numbers will give MemoryError later on).
df_yelp_review = pd.DataFrame()
i=0
for df in df_yelp_review_iter:
    df = df[df['business_id'].isin(df_yelp_business['business_id'])]
    df_yelp_review = pd.concat([df_yelp_review, df])
    i=i+1
    print(i)
    if i==4: break

#Also make sure we only get businesses that already show up in our review list and delete the rest.
    
df_yelp_business = df_yelp_business[df_yelp_business['business_id'].isin(df_yelp_review['business_id'])]
print('Final businesses shape: ', df_yelp_business.shape)
print('Final review shape: ', df_yelp_review.shape)

#preprocessing code
#https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb) which really does a good deal to preprocess the text.    

def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    return text

df_yelp_review['text'] = df_yelp_review['text'].apply(clean_text)

vectorizer_reviews = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
vectorized_reviews = vectorizer_reviews.fit_transform(df_yelp_review['text'])

print(vectorized_reviews.shape) 

' | '.join(vectorizer_reviews.get_feature_names()[:100]) # only the first 100   

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(df_yelp_review['text'])
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

final_tf_idf = tf_idf_vect.transform(df_yelp_review['text'])
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words includin-g both unigrams and bigrams ", final_tf_idf.get_shape()        

vectorizer_categories = CountVectorizer(min_df = 1, max_df = 1., tokenizer = lambda x: x.split(', '))
vectorized_categories = vectorizer_categories.fit_transform(df_yelp_business['categories'])

' | '.join(vectorizer_categories.get_feature_names()[:100]) # only the first 100

print('restuarants x categories: \t', vectorized_categories.shape) 
print('restuarants x reviews: \t\t' , businessxreview.shape) 
print('reviews x words: \t\t', vectorized_reviews.shape)


# to choose a restaurant, just copy the business id and paste it in the next cell
# you can always rerun the cell to choose another restuarant. 

df_yelp_business.sample(10)
business_choose = 'aUrOyWFKxKeVXiFzwbTXSA' # vegan, vegetarian, cafes
new_reviews = df_yelp_review.loc[df_yelp_review['business_id'] == business_choose, 'text']
print('\n'.join([r[:100] for r in new_reviews.tolist()])) # restaurant reviews
#*Then we pull up the categories:
new_categories = df_yelp_business.loc[df_yelp_business['business_id'] == business_choose, 'categories']
new_categories.tolist() #  restaurant categories
#Here, we compute two sets of distancecs: we compute the correlation distance of the average vectorized reviews to all the reviews, and compute the correlation distance between this category and all other categories. The category trick will be clearer when we see the results.


from scipy.spatial.distance import cdist
# find most similar reviews
dists1 = cdist(vectorizer_reviews.transform(new_reviews).todense().mean(axis=0), 
              vectorized_reviews.T.dot(businessxreview).T.todense(), 
               metric='correlation')
# find most similar categories
dists2 = cdist(vectorizer_categories.transform(new_categories).todense().mean(axis=0), 
              vectorized_categories.todense(), 
               metric='correlation')
#*Now we combine the two sets of distances and take the average of those (we can take other metrics such as min or max, depending on your priority).
# combine the two vectors in one matrix
dists_together = np.vstack([dists1.ravel(), dists2.ravel()]).T
# this is a key cell: how are we going to prioritize ?
dists = dists_together.mean(axis=1)
# select the closest 10

closest = dists.argsort().ravel()[:10]
df_yelp_business.loc[df_yelp_business['business_id']== business_choose, ['business_id', 'categories', 'name', 'stars']]
df_yelp_business.loc[df_yelp_business['business_id'].isin(df_yelp_business['business_id'].iloc[closest]), ['business_id', 'categories', 'name', 'stars']]

