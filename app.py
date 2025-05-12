import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis with LLM", layout="wide")
st.title("Data Analysis with LLM")
st.markdown("Upload a dataset to analyze or visualize the data.")
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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dataset Table View")
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        df = load_data(uploaded_file, file_extension)
        if df is not None:
            st.write(df)
            st.sidebar.success("Dataset loaded successfully!")
        else:
            st.sidebar.error("Failed to load dataset.")
    else:
        st.info("No dataset uploaded yet. Upload a file to view the table.")

with col2:
    st.subheader("Data Analyst")
    st.markdown("Specialized in data analysis and visualization")
    st.info("Upload a dataset to start analyzing.")