import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from state_manager import get_liked_books, initialize_state, add_book_to_liked, remove_book_from_liked
import search
import recommend

@st.cache_data
def filter_books(query):
    query = query.lower()
    return search.search_book(query, search.vectorizer)

initialize_state()

def display_books(filtered_df_name, liked_section=False):
    num_columns = 3

    for i in range(0, len(filtered_df_name), num_columns):
        cols = st.columns(num_columns)

        for j, col in enumerate(cols):
            if i + j < len(filtered_df_name):
                row = filtered_df_name.iloc[i + j]
                with col:
                    st.image(row.get('cover_image', ""), width=200)
                    st.markdown(f"**{row.get('title', 'No Title')}**")
                    st.text(f"Ratings: {row.get('ratings', 'N/A')}")
                    
                    if 'num_pages' in row and pd.notna(row['num_pages']):
                        st.text(f"Pages: {row['num_pages']}")
                    
                    st.markdown(f"[More details]({row.get('url', '#')})", unsafe_allow_html=True)

                    # Use unique key by appending index
                    unique_key = f"{row.get('book_id', '')}_{i + j}"

                    if liked_section:
                        if st.button(f"Unlike {row.get('title', 'this book')}", key=f"unlike_button_{unique_key}"):
                            remove_book_from_liked(row['book_id'])
                            st.warning(f"Removed {row.get('title', 'this book')} from liked books")
                    else:
                        if st.button(f"Like {row.get('title', 'this book')}", key=f"like_button_{unique_key}"):
                            add_book_to_liked(row.to_dict())
                            st.success(f"Liked {row.get('title', 'this book')}")

                    st.write("")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Search", "Recommendations", "Liked Books", "TimeSync"],
        icons=["house", "search", "book", "heart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#F0F0F0aa"},
            "nav-link-selected": {"background-color": "#4CAF50"}
        }
    )

if selected == "Home":
    st.title('Book Recommendation System')
    st.write("""
    Welcome to our Book Recommendation System! 
    Dive into a curated list of top recommendations just for you.
    """)
    st.image("./images/wp12065638-white-book-wallpapers.jpg")
    st.write("Happy Reading!")

if selected == "Search":
    search_query_name = st.text_input("Enter book name:")
    if 'search_query_name' not in st.session_state:
        st.session_state.search_query_name = ''

    search_button_name = st.button("Search")
    if search_button_name:
        st.session_state.search_query_name = search_query_name

    filtered_df_name = filter_books(st.session_state.search_query_name)
    if st.session_state.search_query_name and filtered_df_name.empty:
        st.write("No books found. Please try a different search term.")
    elif st.session_state.search_query_name:
        display_books(filtered_df_name)

elif selected == "Recommendations":
    st.write("Similar Books")
    content_df = pd.read_csv('./liked_books.csv')
    content_book_ids = content_df['book_id']

    for book_id in content_book_ids:
        display_books(search.recommend_content(str(book_id)))

    st.write("Books for you-")
    display_books(recommend.recommend_book(st.session_state.liked_books))

elif selected == "Liked Books":
    st.write("Liked Books")
    liked_books_df = get_liked_books()
    display_books(liked_books_df, liked_section=True)

elif selected == "TimeSync":
    st.title('TimeSync - Adaptive Book Suggestions by Time Availability')
    st.write("Short on Time?")
    st.write("Explore new books you can read within your available time.")
    input_time = st.number_input('Enter time you have in minutes.', min_value=0, max_value=1000, value=50, step=1)
    get_time_button = st.button("Find")

    if get_time_button:
        df_time = search.recommmend_books_by_time(input_time)
        display_books(df_time)
