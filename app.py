from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import ast, difflib, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors



model = joblib.load('kmeans_model.pkl') 
vectorizer = joblib.load('vectorizer.pkl') 
using_df = pd.read_csv('usebooks.csv')

using_df['text'] = using_df['title'].fillna('') + ' ' + using_df['description'].fillna('') + ' ' + using_df['author'].fillna('') + ' ' + using_df['categorias'].fillna('')
using_df['cluster'] = model.predict(vectorizer.transform(using_df['text']))
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form['title'].strip()

    if not query:
        return jsonify({"error": "No title provided"})
    query_vec = vectorizer.transform([query])
    cluster_label = model.predict(query_vec)[0]
    cluster_books = using_df[using_df['cluster'] == cluster_label]
    cluster_vectors = vectorizer.transform(cluster_books['text'])
    similarities = (cluster_vectors * query_vec.T).toarray().flatten()
    top_indices = similarities.argsort()[::-1][:5]  
    recommendations = []
    for idx in top_indices:
        book = cluster_books.iloc[idx]
        recommendations.append({
            "title": book['title'],
            "author": book['author'] if pd.notna(book['author']) else "Unknown",
            "description": book['description'] if pd.notna(book['description']) else "No description available",
            "categories": book['categorias'] if pd.notna(book['categorias']) else "Unknown",
            "similarity": similarities[idx]
        })

    print(recommendations)
    return jsonify(recommendations)





if __name__ == '__main__':
    app.run(debug=True)