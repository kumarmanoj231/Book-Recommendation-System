import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Cache the csv_book_mapping to avoid repeated file reads
@st.cache_data
def load_book_mapping():
    csv_book_mapping = {}
    with open("book_id_map.csv", "r") as f:
        for line in f:
            csv_id, book_id = line.strip().split(",")
            csv_book_mapping[csv_id] = book_id
    return csv_book_mapping

# Load books and mapping files only once
csv_book_mapping = load_book_mapping()

# Load books titles only once
@st.cache_data
def load_books_titles():
    books_titles = pd.read_json("books_titles.json")
    books_titles["book_id"] = books_titles["book_id"].astype(str)
    return books_titles

def recommend_book(liked_books):
    my_books = pd.read_csv("liked_books.csv", index_col=0)
    my_books["book_id"] = my_books["book_id"].astype(str)
    my_books = my_books[['user_id','book_id','rating','title']]

    # Set up book set to match books with users
    book_set = set(my_books["book_id"])

    # Step 1: Find overlapping users by reading goodreads_interactions file line by line
    overlap_users = {}
    with open("goodreads_interactions.csv", 'r') as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.split(",")
            book_id = csv_book_mapping.get(csv_id)
            if book_id in book_set:
                overlap_users[user_id] = overlap_users.get(user_id, 0) + 1

    # Filter overlap users based on threshold
    threshold = my_books.shape[0] / 5
    filtered_overlap_users = {k for k, v in overlap_users.items() if v > threshold}

    # Step 2: Filter interactions for recommendations (again, read the file line by line)
    interactions_list = []
    with open("goodreads_interactions.csv", 'r') as f:
        for line in f:
            user_id, csv_id, _, rating, _ = line.split(",")
            if user_id in filtered_overlap_users:
                book_id = csv_book_mapping.get(csv_id)
                if book_id:
                    interactions_list.append([user_id, book_id, rating])

    # Create interactions DataFrame and filter out empty or all-NA columns
    my_books_filtered = my_books[["user_id", "book_id", "rating"]].dropna(how='all', axis=1)
    interactions_filtered = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"]).dropna(how='all', axis=1)

    # Concatenate the filtered DataFrames
    interactions = pd.concat([my_books_filtered, interactions_filtered])

    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"])
    interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
    interactions["book_index"] = interactions["book_id"].astype("category").cat.codes

    # Build sparse matrix for cosine similarity
    ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
    ratings_mat = ratings_mat_coo.tocsr()

    # Compute similarity with the current user (my_index assumed to be 0)
    my_index = 0
    similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()

    # Select top 15 similar users
    indices = np.argpartition(similarity, -15)[-15:] if len(similarity) > 15 else np.arange(len(similarity))
    similar_users = interactions[interactions["user_index"].isin(indices)].copy()
    similar_users = similar_users[similar_users["user_id"] != "-1"]

    # Aggregate recommendations
    book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
    books_titles = load_books_titles()
    book_recs = book_recs.merge(books_titles, how="inner", on="book_id")

    # Adjust and score the recommendations
    book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["ratings"])
    book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]
    book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]

    # Filter based on similarity and user rating thresholds
    my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", " ", regex=True)
    book_recs = book_recs[~book_recs["mod_title"].isin(my_books["mod_title"])]
    book_recs = book_recs[book_recs["mean"] >= 4]
    book_recs = book_recs[book_recs["count"] > 2]

    # Select top 10 recommendations
    top_recs = book_recs.sort_values("mean", ascending=False).head(10) if len(book_recs) > 20 else book_recs

    return top_recs
