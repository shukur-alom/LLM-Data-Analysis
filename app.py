import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
import os
from pandasai import SmartDataframe
from pandasai.exceptions import NoCodeFoundError


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
def chatbot_fallback(llm, question, columns):
    columns_str = ", ".join(columns)
    fallback_prompt = f"""
    You are a data analysis assistant with a dataset containing columns: {columns_str}.
    The query '{question}' failed.
    Suggest a PandasAI-compatible analysis or visualization using 1-2 relevant columns (e.g., 'plot a bar chart of {columns[0]} by {columns[1]}').
    """
    try:
        return llm.invoke(fallback_prompt).content.strip()
    except Exception as e:
        return f"Fallback response error: {e}"
    
def improve_prompt(llm, original_prompt, columns):
    columns_str = ", ".join(columns)
    meta_prompt = f"""
    You are an expert data analyst using PandasAI to analyze a dataset with columns: {columns_str}.
    The user's query is: '{original_prompt}'.
    Rewrite the query to be concise, specific, and optimized for PandasAI (e.g., 'plot a bar chart of sales by region').
    Return only the rewritten query as a single sentence.
    """
    try:
        return llm.invoke(meta_prompt).content.strip()
    except Exception as e:
        st.warning(f"Prompt improvement failed: {e}")
        return original_prompt

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

def is_visualization(answer):
    if not isinstance(answer, str) and answer is not None:
        return True
    if isinstance(answer, str):
        keywords = ["chart", "plot", "graph", "figure", "visualization"]
        return any(keyword in answer.lower() for keyword in keywords)
    return False

with col2:
    st.subheader("Data Analyst")
    st.markdown("Specialized in data analysis and visualization")
    if uploaded_file:
        llm = init_llm()
        if llm:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            df = load_data(uploaded_file, file_extension)
            if df is not None:
                smart_df = SmartDataframe(df, config={"llm": llm, "enable_cache": False})
                user_input = st.text_input("Ask me anything about your data (e.g., 'Plot sales by region')", "")
                if st.button("Analyze"):
                    if user_input:
                        with st.spinner("Processing your query..."):
                            answer = smart_df.chat(user_input)
                            if is_visualization(answer):
                                image_path = 'exports/charts/temp_chart.png'
                                if os.path.exists(image_path):
                                    st.image(image_path, caption="Generated Visualization")
                                else:
                                    st.warning("Visualization generated but could not be displayed.")
                            else:
                                st.write("**Result:**", answer)
                    else:
                        st.warning("Please enter a query to analyze.")
                else:
                    st.info("Enter a query to analyze the dataset.")
            else:
                st.error("Failed to load dataset.")
        else:
            st.error("Failed to initialize LLM.")
    else:
        st.info("Upload a dataset to start analyzing.")


# ... (add to analysis block in Data Analyst section)
if st.button("Analyze"):
    if user_input:
        with st.spinner("Processing your query..."):
            try:
                answer = smart_df.chat(user_input)
                if is_visualization(answer):
                    image_path = 'exports/charts/temp_chart.png'
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Generated Visualization")
                    else:
                        st.warning("Visualization generated but could not be displayed.")
                else:
                    st.write("**Result:**", answer)
            except NoCodeFoundError as e:
                st.warning(f"PandasAI error: {e}. Please try a different query.")
            except Exception as e:
                st.warning(f"Error: {e}. Please try a different query.")
    else:
        st.warning("Please enter a query to analyze.")