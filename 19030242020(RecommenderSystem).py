# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 01:27:16 2019

@author: Rajesh
"""

# Import Pandas
import pandas as pd
    
metadata = pd.read_csv("C:/Users/Rajesh\Desktop/flipkart_com-ecommerce_sample.csv")

# Print the first three rows
metadata.head(3)
metadata = metadata.drop(columns="pid")
metadata = metadata.drop(["crawl_timestamp", "product_url", "image","is_FK_Advantage_product","product_rating","overall_rating"], axis=1)
metadata=metadata.drop(columns="product_specifications")
metadata.head()

# Function that computes the discount percentage
def discount_percent(x):
    V = x['retail_price']
    R = x['discounted_price']
    # Calculation 
    return ((V-R)/V )* 100

# Define a new column 'DiscountPercent' having calculated value with `discount_percent()` function
metadata['DiscountPercent'] = metadata.apply(discount_percent, axis=1)
metadata.head()

#Sort products based on discount % calculated above
metadata = metadata.sort_values('DiscountPercent', ascending=False)

#Print the top 10 top discounted products
metadata[['product_name', 'description', 'DiscountPercent']].head(10)



# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:02:51 2019

@author: Rajesh
"""


import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 100)
df = pd.read_csv("C:/Users/Rajesh\Desktop/flipkart_com-ecommerce_sample.csv")
df.head()

df.shape

df = df.drop(["uniq_id","retail_price","discounted_price","crawl_timestamp","pid", "product_url", "image","is_FK_Advantage_product","product_rating","overall_rating","product_specifications"], axis=1)
df.head()

df.shape

df.dtypes

df['product_category_tree'] = df['product_category_tree'].astype('str') 
df['description'] = df['description'].astype('str')
df['brand'] = df['brand'].astype('str')

df.dtypes

# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    description = row['description']
    
    # instantiating Rake, by default is uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(str(description))

    # getting the dictionary whith key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column
    row['Key_words'] = list(key_words_dict_scores.keys())
    
    
df.set_index('product_name', inplace = True)
df.head()


df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if (col != 'description','brand'):
           words = words + ''.join(row[col])+ ' '
        else:
          words = words + row[col]+ ' '
    row['bag_of_words'] = words


    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)
df.head()


# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the product nameso they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices[:5]

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim



# function that takes in product name as input and returns the top 10 recommended products
def recommendations(product_name, cosine_sim = cosine_sim):
    
    recommended_products = []
    
    # gettin the index of the product that matches the title
    idx = indices[indices == product_name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar products
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the names of the best 10 matching products
    for i in top_10_indexes:
        recommended_products.append(list(df.index)[i])
        
    return recommended_products

recommendations('AW Bellies')