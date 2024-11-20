from flask import Flask, render_template, jsonify, request
import pandas as pd
import ast, difflib

app = Flask(__name__)

final_df = pd.read_csv('final_books_df.csv')
similarity_df = pd.read_csv('similarity_books_scores.csv')
raw_books_df = pd.read_csv('raw_books.csv')

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title').lower()  

    all_titles = final_df['title'].str.lower().tolist()
    closest_matches = difflib.get_close_matches(title, all_titles, n=5, cutoff=0.5) 
    print(f"Closest matches: {closest_matches}")
    
    if not closest_matches:
        return jsonify({'error': 'No matching titles found'}), 400

    title_index_pairs = []
    for match in closest_matches:
        score = difflib.SequenceMatcher(None, title, match).ratio()  
        index = final_df[final_df['title'].str.lower() == match].index[0]  
        title_index_pairs.append((match, index, score))


    title_index_pairs.sort(key=lambda x: x[2], reverse=True)

    matched_title, matched_index, matched_score = title_index_pairs[0]
    similarity_scores = similarity_df.iloc[matched_index].tolist()

    similarity_pairs = [(i, similarity_scores[i]) for i in range(len(similarity_scores))]
    sorted_similarity_pairs = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)

    top_5_similar = sorted_similarity_pairs[:5] 
    top_5_similar_indexes = [pair[0] for pair in top_5_similar]

    top_5_similar_titles = final_df.iloc[top_5_similar_indexes]['title'].tolist()
    print(f"Top 5 Similar titles: {top_5_similar_titles}")
    recommended_books = []

    def handle_missing(value):
        return value if pd.notna(value) else 'Missing'

    for title in closest_matches:

        matched_book = raw_books_df[raw_books_df['title'].str.lower() == title.lower()]
        
        if not matched_book.empty:
            book_info = matched_book.iloc[0]
            recommended_books.append({
                'title': handle_missing(book_info['title']),
                'description': handle_missing(book_info['description']),
                'author': handle_missing(book_info['author']),
                'isbn10': handle_missing(book_info['isbn10']),
                'isbn13': handle_missing(book_info['isbn13']),
                'publish_date': handle_missing(book_info['publish_date']),
                'edition': handle_missing(book_info['edition']),
                'best_seller': handle_missing(book_info['best_seller']),
                'top_rated': handle_missing(book_info['top_rated ']),  # trash formatting trail space as is
                'rating': handle_missing(book_info['rating']),
                'review_count': handle_missing(book_info['review_count']),
                'price': handle_missing(book_info['price']),
            })

    for title in top_5_similar_titles:
        matched_book = raw_books_df[raw_books_df['title'].str.lower() == title.lower()]

        if not matched_book.empty:
            book_info = matched_book.iloc[0]
            recommended_books.append({
                'title': handle_missing(book_info['title']),
                'description': handle_missing(book_info['description']),
                'author': handle_missing(book_info['author']),
                'isbn10': handle_missing(book_info['isbn10']),
                'isbn13': handle_missing(book_info['isbn13']),
                'publish_date': handle_missing(book_info['publish_date']),
                'edition': handle_missing(book_info['edition']),
                'best_seller': handle_missing(book_info['best_seller']),
                'top_rated': handle_missing(book_info['top_rated ']),  #shit formatting lmao
                'rating': handle_missing(book_info['rating']),
                'review_count': handle_missing(book_info['review_count']),
                'price': handle_missing(book_info['price']),
            })

    print(f"Recommended Books Data: {recommended_books}")
    return jsonify(recommended_books)





if __name__ == '__main__':
    app.run(debug=True)