import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
import os

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


file_path = "amazon_co-ecommerce_sample.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        print("The file is empty.")
    else:
        # Remove unnecessary columns
        df.drop(columns = ['uniq_id', 'manufacturer', 'price', 
                            'number_available_in_stock', 'number_of_reviews', 
                            'number_of_answered_questions', 'average_review_rating',
                            'items_customers_buy_after_viewing_this_item',
                            'customer_questions_and_answers', 'customer_reviews', 'sellers'], axis=1, inplace = True)

        # Remove NaN values
        df.dropna(inplace=True)
        print(df.head())
else:
    print("File not found!")


# Define tokenizer and stemmer
ps = PorterStemmer()

def tokenize_stem(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove digits and special characters except whitespace
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text word-wise
    tokens = nltk.word_tokenize(text)
    
    # Keep only base form of words
    stemmed_tokens = [ps.stem(token) for token in tokens]
    
    # Return words joined by space
    return " ".join(stemmed_tokens)

# Create stemmed tokens column
df['stemmed_tokens'] = df.apply(lambda row: tokenize_stem(row['product_name'] + " " + row['amazon_category_and_sub_category'] + " " + row['description']), axis = 1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf = TfidfVectorizer()

def similarity(text1, text2):
  matrix = tfidf.fit_transform([text1, text2])
  return cosine_similarity(matrix)[0][1]

# Define recommend function
def recommend(product_name):
  stemmed_name = tokenize_stem(product_name)

  df['similarity'] = df['stemmed_tokens'].apply(lambda x: similarity(x, stemmed_name))
  
  results = df.sort_values(by='similarity', ascending=False).head(10)
    
  # Select relevant columns, including similarity
  return results[['similarity','product_name', 'amazon_category_and_sub_category', 'description']]

# Create Streamlit app
st.image('download.jpeg')
st.title('Amazon Product Search')

# Create search box and button
query = st.text_input('Enter a product name')
search_button = st.button('Search')

# Perform search and display results
if search_button:
    results = recommend(query)
    st.write(results)
