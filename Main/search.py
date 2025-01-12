import gzip
import pandas as pd
import json


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def parse_fields(line):
    data = json.loads(line)
    return{
        "book_id" : data['book_id'],
        "title" : data['title_without_series'],
        "ratings" : data['ratings_count'],
        "url" : data['url'],
        "cover_image" : data['image_url'],
        "num_pages" : data["num_pages"],
        "similar_books" : data["similar_books"],
    }

book_titles = []
with gzip.open("goodreads_books.json.gz","r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        
        fields = parse_fields(line)

        try:
            ratings = int(fields['ratings'])
            num_pages = int(fields['num_pages'])
        except ValueError:
            continue
        if ratings > 15 :
            book_titles.append(fields)

titles = pd.DataFrame.from_dict(book_titles)
titles['ratings'] = pd.to_numeric(titles['ratings'])
titles['num_pages'] = pd.to_numeric(titles['num_pages'])

titles['mod_title'] = titles['title'].str.replace("[^a-zA-Z0-9 ]","",regex=True)

titles = titles[titles['mod_title'].str.len() >0]

# titles.to_json("./books_titles.json")

df = titles[titles['num_pages'] > 3]

tfidf = vectorizer.fit_transform(titles["mod_title"])


def search_book(query,vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -20)[-20:]
    results = titles.iloc[indices]
    results = results.sort_values("ratings", ascending=False)
    results.drop('mod_title',axis=1,inplace=True)
    return results


def pages_read_in_time(time_in_minutes, reading_speed_words_per_minute=225, words_per_page=275):
    total_words_read = time_in_minutes * reading_speed_words_per_minute
    pages_read = total_words_read / words_per_page
    return pages_read

def recommmend_books_by_time(time_in_minutes):
    pages = pages_read_in_time(time_in_minutes)
    return df[df['num_pages'] <=pages].sort_values(by=['num_pages','ratings'],ascending=False).head(20)
    


def recommend_content(book_id):
    # Check if the book exists in the titles DataFrame
    if book_id not in titles['book_id'].values:
        print(f"Book ID {book_id} not found.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Retrieve the book's row
    book = titles.loc[titles['book_id'] == book_id]

    # If the book DataFrame is empty, return early
    if book.empty:
        print(f"No book data found for Book ID {book_id}.")
        return pd.DataFrame()

    # Extract similar books
    similar_books = book['similar_books'].iloc[0]  # Access the first (and only) row

    # Ensure similar_books is a valid list or array
    if not isinstance(similar_books, (list, np.ndarray)):
        print(f"Invalid 'similar_books' format for Book ID {book_id}.")
        return pd.DataFrame()

    # Filter titles for similar books
    similar_books_df = titles[titles['book_id'].isin(similar_books)]

    # Define desired columns
    columns = ["book_id", "title", "ratings", "url", "cover_image", "num_pages", "similar_books"]

    # Return the filtered DataFrame with selected columns
    return similar_books_df[columns]
