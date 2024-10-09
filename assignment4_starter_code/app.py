from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__) 


# TODO: Fetch dataset, initialize vectorizer and LSA here

# fetched dataset?? 
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')

# Step 2: Initialize vectorizer and transform the data into TF-IDF representation
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(newsgroups.data)

# Step 3: Implement LSA
def perform_lsa(tfidf_matrix, n_components):
    # Perform SVD
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)  # This applies SVD to the term-document matrix
    
    # Get the explained variance
    explained_variance = svd.explained_variance_ratio_
    
    return lsa_matrix, explained_variance, svd

# Choose the number of topics/components
n_topics = 10  # or based on your analysis
lsa_matrix, explained_variance, svd = perform_lsa(tfidf_matrix, n_topics)

def search_engine(query, vectorizer, svd, lsa_matrix, top_n=5):
    """
    Function to search for top 5 similar documents given a query
    Input: 
        - query (str): User's search query
        - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer
        - svd (TruncatedSVD): The fitted SVD model
        - lsa_matrix (np.ndarray): The LSA representation of documents
        - top_n (int): Number of top documents to return
    Output: 
        - documents (list): List of top similar documents
        - similarities (list): List of similarity scores
        - indices (list): List of indices of the top documents
    """
    # Step 1: Transform the query to TF-IDF representation
    query_tfidf = vectorizer.transform([query])  # Transform the query to TF-IDF space
    
    # Step 2: Project the query into LSA space
    query_lsa = svd.transform(query_tfidf)  # Transform the query using the SVD model
    
    # Step 3: Calculate cosine similarity between query and all documents in LSA space
    similarities = cosine_similarity(query_lsa, lsa_matrix)  # Compute cosine similarity
    similarities = similarities.flatten()  # Flatten the array for easy indexing
    
    # Step 4: Get indices of the top N similar documents
    indices = np.argsort(similarities)[-top_n:][::-1]  # Get indices of top N similar documents
    
    # Retrieve the documents and their similarity scores
    documents = [newsgroups.data[i] for i in indices]  # Extract the top documents
    similarities = similarities[indices]  # Get corresponding similarity scores
    
    return documents, similarities.tolist(), indices.tolist() 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query, vectorizer, svd, lsa_matrix)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
