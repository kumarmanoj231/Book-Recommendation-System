import streamlit as st
import pandas as pd
import os

LIKED_BOOKS_FILE = 'liked_books.csv'

def initialize_state():
    if 'liked_books' not in st.session_state:
        if os.path.exists(LIKED_BOOKS_FILE):
            st.session_state.liked_books = pd.read_csv(LIKED_BOOKS_FILE)
        else:
            st.session_state.liked_books = pd.DataFrame(columns=["user_id", "book_id", "ratings", "title", "cover_image", "url", "num_pages"])
            st.session_state.liked_books.to_csv(LIKED_BOOKS_FILE, index=False)

def add_book_to_liked(book_data):
    initialize_state()
    if not any(st.session_state.liked_books['book_id'] == book_data['book_id']):
        new_book = {
            "user_id": -1,
            "book_id": book_data['book_id'],
            "rating": book_data.get('ratings', 5),
            "title": book_data.get('title', 'Unknown'),
            "cover_image": book_data.get('cover_image', ''),
            "url": book_data.get('url', ''),
            "num_pages": book_data.get('num_pages', '')
        }
        st.session_state.liked_books = pd.concat([st.session_state.liked_books, pd.DataFrame([new_book])], ignore_index=False)
        st.session_state.liked_books.to_csv(LIKED_BOOKS_FILE, index=True)

def remove_book_from_liked(book_id):
    initialize_state()
    st.session_state.liked_books = st.session_state.liked_books[st.session_state.liked_books['book_id'] != book_id]
    st.session_state.liked_books.to_csv(LIKED_BOOKS_FILE, index=False)

def get_liked_books():
    initialize_state()
    return st.session_state.liked_books
