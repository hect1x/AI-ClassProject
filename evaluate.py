import pandas as pd
import joblib
import numpy as np

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
using_df = pd.read_csv('usebooks.csv')

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

def evaluate_query(query):
    query_vec = vectorizer.transform([query])
    distances, indices = model.kneighbors(query_vec)
    top_books = using_df.iloc[indices[0]]
    top_similarities = 1 - distances[0] 
    
    results = []
    print(f"Query: {query}")
    for i, (book, sim) in enumerate(zip(top_books[['title', 'description', 'author', 'categorias']].values, top_similarities)):
        book_details = {
            "title": book[0],
            "description": book[1],
            "author": book[2],
            "categorias": book[3],
            "similarity": sim
        }
        results.append(book_details)
        print(f"{i+1}. {book[0]} (Similarity: {sim:.4f})")
    # print(results)
    return results

all = []
for query in queries:
    similarities = evaluate_query(query)
    for result in similarities:
        all.append(result['similarity'])

average = np.mean(all)
print(f"\nAverage Cosine Similarity for all queries: {average}")
