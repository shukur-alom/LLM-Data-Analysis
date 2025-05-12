import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis with LLM", layout="wide")
st.title("Data Analysis with LLM")
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])

def load_data(uploaded_file, file_type):
    try:
        if file_type == "csv":
            return pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    df = load_data(uploaded_file, file_extension)
    if df is not None:
        st.write("Dataset Table View")
        st.write(df)
        st.sidebar.success("Dataset loaded successfully!")
    else:
        st.sidebar.error("Failed to load dataset.")
else:
    st.info("Upload a CSV or Excel file to view the dataset.")