import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
using_df = pd.read_csv('usebooks.csv')
using_df['text'] = using_df['title'].fillna('') + ' ' + using_df['description'].fillna('') + ' ' + using_df['author'].fillna('') + ' ' + using_df['categorias'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(using_df['text'])

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
score = 0
def evaluate_query(query):
    global score
    query_vec = vectorizer.transform([query])
    
    predicted_cluster = kmeans.predict(query_vec)[0]
    
    cluster_books = using_df[kmeans.labels_ == predicted_cluster]
    
    book_vectors = vectorizer.transform(cluster_books['text'])
    similarities = cosine_similarity(query_vec, book_vectors).flatten()
    
    cluster_books['similarity'] = similarities
    sorted_books = cluster_books.sort_values(by='similarity', ascending=False)
    
    top_books = sorted_books[['title', 'description', 'author', 'categorias', 'similarity']].head(5)
    temp = 0
    print(f"Query: {query}")
    for idx, row in top_books.iterrows():
        print(f"Similarity: {row['similarity']:.4f}\n")
        if isinstance(row['similarity'], (int, float)):
            temp += row['similarity']
    
    score += temp/5
    return similarities

queries = [
    "Programming for beginners",
    "How do I program in C++?",
    "Which database should I choose in Amazon RDS?",
    "Best books for software engineering",
    "Java or Javascript tutorials",
    "How to get started with web programming?",
    "What are the best machine learning resources?",
    "Books on algorithms and data structures",
    "Introduction to Python for data science",
    "Books for learning web development"
]

all_similarities = []
for query in queries:
    similarities = evaluate_query(query)
    all_similarities.extend(similarities)

average_similarity = score/len(queries)
print(f"\nAverage Cosine Similarity for all queries: {average_similarity:.4f}")
