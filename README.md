# 256Individualproject
# Yelp Restaurant Recommendation system   

##### Name: Sai Sarath Vattikuti     
##### SJSU ID: 013821483
##### course: CMPE 256 Summer-2019
##### Instructo: Shih Yu Chang

# Abstract:
The model built in this project will recommend restaurants to the users based upon the current ratings of the user.Unlike the similarity based recommendation, this model in this project recommends unique restaurants to the users which are worth exploring.The data for this model is scraped from the yelp website using python scripting.The data collected is cleaned and preprocessed and then the model is applied to the preprocessed data for recommending surprisingly new restaurants which user may like 

# Workflow:
First, we want to find a way to represent reviews using a bag-of-words and Tf-Idf representation. After doing so, we will also represent categories using a one-hot encoding representation.
Then, we can manipulate those representations to find similarities and differences while
balancing the weights of the two.
The core idea assumes that you are more likely to love a restaurant if its reviews are similar to the reviews of the restaurants you already love.



Building model:
# vectorization(Bag of words):
Now the review and categories are vectorized using count vectorizer method in python which is simple Bag Of Words.

 
# TF-IDF :
Term frequency is calculated by counting the number of times each word is repeated in the given document divided to the number of words in the given document. Inverse document 
Frequency is called as log of the number of documents present divided to the documents containing the word


                                              


# Files
The data which is webscraped is uploaded in the Data folder
The code for webscraping is also uploaded i a different file
The code for the model I have built is also uploaded in a different file 

[DATASET](https://github.com/SaiSarathVattikuti/project/blob/master/yelp_academic_dataset_business.json.zip)<br>
[SCRAPING CODE](https://github.com/SaiSarathVattikuti/project/blob/master/webscraper.py)<br>
[SOURCE CODE](https://github.com/SaiSarathVattikuti/project/blob/master/project_code.py)<br>
[REPORT](https://github.com/SaiSarathVattikuti/project/blob/master/Yelp%20Recommendation%20System.pdf)
