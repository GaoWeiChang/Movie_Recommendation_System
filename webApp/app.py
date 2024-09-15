import streamlit as st
from recommend import Recommend

# Tab name and icon
st.set_page_config(
    page_title="Movie Recommendation System", 
    page_icon="🎬",  
)

Recommend()