import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis with LLM", layout="wide")
st.title("Data Analysis with LLM")
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Table View")
        st.write(df)
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
else:
    st.info("Upload a CSV file to view the dataset.")