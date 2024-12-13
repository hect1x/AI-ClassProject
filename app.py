from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import ast, difflib, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors



model = joblib.load('model.pkl') 
vectorizer = joblib.load('vectorizer.pkl') 
using_df = pd.read_csv('usebooks.csv')

using_df['text'] = using_df['title'].fillna('') + ' ' + using_df['description'].fillna('') + ' ' + using_df['author'].fillna('') + ' ' + using_df['categorias'].fillna('')

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

    distances, indices = model.kneighbors(query_vec)
    recommendations = []
    
    for idx, dist in zip(indices[0], distances[0]):
        book = using_df.iloc[idx]
        recommendations.append({
            "title": book['title'],
            "author": book['author'],
            "description": book['description'],
            "categories": book['categorias'],  
            "similarity": 1 - dist  
        })

    print(recommendations)
    return jsonify(recommendations)






if __name__ == '__main__':
    app.run(debug=True)