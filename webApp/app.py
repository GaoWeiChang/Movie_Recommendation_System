import streamlit as st

# Tab name and icon
st.set_page_config(
    page_title="Movie Recommendation System", 
    page_icon="🎬",  
)

# for execute 
page = st.sidebar.selectbox('Navigate', ("Recommended Movies", "Resources"))

if page == "Recommended Movies":
    summarize_page()
elif page == "Resources":
    resources_page()