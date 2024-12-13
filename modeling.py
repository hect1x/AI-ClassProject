import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

using_df = pd.read_csv('usebooks.csv')

using_df['text'] = using_df['title'].fillna('') + ' ' + using_df['description'].fillna('') + ' ' + using_df['author'].fillna('') + ' ' + using_df['categorias'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(using_df['text'])

model = NearestNeighbors(n_neighbors=5, metric='cosine')

model.fit(x)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

