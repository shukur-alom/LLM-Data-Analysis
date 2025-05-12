import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
import os

st.set_page_config(page_title="Data Analysis with LLM", layout="wide")
st.title("Data Analysis with LLM")
st.markdown("Upload a dataset to analyze or visualize the data.")
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])

def init_llm():
    try:
        return ChatGroq(model_name="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
    except KeyError:
        st.error("Error: GROQ_API_KEY environment variable not set.")
        return None
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

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
    if uploaded_file:
        llm = init_llm()
        if llm:
            st.success("LLM initialized successfully!")
        else:
            st.error("Failed to initialize LLM.")
    else:
        st.info("Upload a dataset to start analyzing.")