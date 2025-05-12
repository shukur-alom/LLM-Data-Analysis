import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
import os
from pandasai import SmartDataframe
from pandasai.exceptions import NoCodeFoundError
import sqlite3

# Streamlit page configuration
st.set_page_config(page_title="Data Analysis with LLM", layout="wide")

def init_llm():
    """Initialize the Groq LLM."""
    try:
        return ChatGroq(model_name="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
    except KeyError:
        st.error("Error: GROQ_API_KEY environment variable not set.")
        return None
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def improve_prompt(llm, original_prompt, columns):
    """Refine the prompt for better visualization or clarity using dataset columns."""
    columns_str = ", ".join(columns)
    meta_prompt = f"""
    You are an expert data analyst using PandasAI to analyze a dataset with columns: {columns_str}.
    The user's query is: '{original_prompt}'.
    Rewrite the query to meet these criteria:
    1. Be concise, specific, and optimized for PandasAI data analysis or visualization (e.g., bar chart, line plot, scatter plot, table).
    2. Explicitly reference one or more relevant columns from the dataset to address the query.
    3. Specify a clear, actionable output, such as a visualization (e.g., 'plot a bar chart') or aggregation (e.g., 'calculate the average').
    4. Replace vague terms (e.g., 'show trends') with precise instructions (e.g., 'plot a line chart of [column] over [time_column]').
    5. Choose a visualization type that suits the query and data context (e.g., line chart for time series, bar chart for comparisons, scatter plot for relationships).
    6. Ensure the query is executable by PandasAI and avoids overly complex or ambiguous phrasing.
    Example:
    - Original: "Show population trends"
    - Rewritten: "Plot a line chart of population by year for each country"
    Return only the rewritten query as a single sentence.
    """
    try:
        return llm.invoke(meta_prompt).content.strip()
    except Exception as e:
        st.warning(f"Prompt improvement failed: {e}")
        return original_prompt

def chatbot_fallback(llm, question, columns):
    """Provide a fallback chatbot response using dataset columns."""
    columns_str = ", ".join(columns)
    fallback_prompt = f"""
    You are a data analysis assistant using PandasAI with a dataset containing columns: {columns_str}.
    The user's query is: '{question}'.
    Provide a concise response (2-3 sentences) that:
    1. Addresses the query using general data analysis principles if specific data is inaccessible.
    2. Suggests 1-2 relevant columns from the dataset to analyze the query effectively.
    3. Recommends a specific, PandasAI-compatible analysis or visualization (e.g., bar chart, line plot, summary table).
    4. If the query is vague, briefly explain why and suggest a precise, actionable query using the available columns.
    Example:
    - Query: "What about population?"
    - Response: "The query is vague. To analyze population, plot a bar chart of population by country or calculate the average population by year. Relevant columns: country, population, year."
    Ensure the response is clear, actionable, and tailored to PandasAI's capabilities (e.g., simple visualizations, aggregations).
    """
    try:
        response = llm.invoke(fallback_prompt).content.strip()
        # Validate that the response references at least one column
        if not any(col in response for col in columns):
            return f"The query '{question}' is unclear. Try plotting a bar chart of {columns[0]} by {columns[1] if len(columns) > 1 else columns[0]}. Relevant columns: {columns_str}."
        return response
    except Exception as e:
        return f"Fallback response error: {e}"

def llm_direct_chat(llm, question):
    """Directly chat with the LLM for a general response."""
    try:
        prompt = f"""
        You are a helpful assistant. The user asked: '{question}'.
        Provide a clear, concise, and informative response based on general knowledge.
        If the question is about data analysis, suggest a general approach without assuming access to a specific dataset.
        """
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error in direct LLM response: {e}"

def is_visualization(answer):
    """Check if the response contains a visualization."""
    if not isinstance(answer, str) and answer is not None:
        return True
    if isinstance(answer, str):
        keywords = ["chart", "plot", "graph", "figure", "visualization"]
        return any(keyword in answer.lower() for keyword in keywords)
    return False

def is_meaningful_text(answer):
    """Check if the response contains meaningful text output, excluding PandasAI error messages."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    error_indicators = [
        "unfortunately, i was not able to answer your question",
        "no code found in the response",
        "error"
    ]
    return not any(indicator in answer.lower() for indicator in error_indicators)

def load_data(uploaded_file, file_type, table_name=None):
    """Load data from uploaded file based on file type."""
    try:
        if file_type == "csv":
            return pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            return pd.read_excel(uploaded_file)
        elif file_type == "db":
            if not table_name:
                raise ValueError("Table name is required for SQLite database files.")
            conn = sqlite3.connect(uploaded_file)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    # Sidebar for dataset upload
    st.sidebar.title("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx", "db"])
    table_name = None

    # Load dataset
    df = None
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "db":
            table_name = st.sidebar.text_input("Enter the table name for SQLite database", "")
            if table_name:
                try:
                    conn = sqlite3.connect(uploaded_file)
                    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                    conn.close()
                    st.sidebar.success("Dataset loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error loading SQLite file: {e}")
        else:
            try:
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    df = pd.read_excel(uploaded_file)
                st.sidebar.success("Dataset loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    # Main content with two columns
    st.title("Data Analysis with LLM")
    st.markdown("Upload a dataset and enter a query to analyze or visualize the data.")

    # Create two columns for table view and analysis
    col1, col2 = st.columns([2, 1])

    # Left column: Dataset table view
    with col1:
        st.subheader("Dataset Table View")
        if df is not None:
            st.write(df)
        else:
            st.info("No dataset uploaded yet. Upload a file to view the table.")

    # Right column: Data Analyst section
    with col2:
        st.subheader("Data Analyst")
        st.markdown("Specialized in data analysis and visualization")
        
        if df is None:
            st.info("No messages yet. Upload a dataset to start analyzing your data with AI.")
        else:
            # Initialize LLM
            llm = init_llm()
            if not llm:
                return

            # Initialize SmartDataframe
            try:
                smart_df = SmartDataframe(df, config={
                    "llm": llm,
                    "enable_cache": False,
                    "verbose": True,
                    "enable_plot": False,
                    "custom_config": {
                        "matplotlib": {
                            "figsize": (96, 64)
                        }
                    }
                })
            except Exception as e:
                st.error(f"Error initializing SmartDataframe: {e}")
                return

            # Query input
            user_input = st.text_input("Ask me anything about your data and visualizations (e.g., 'Plot population by country')", "")
            if not user_input:
                st.info("Please enter a query to analyze the dataset.")
            
            image_path = 'exports/charts/temp_chart.png'
            # Process query
            if st.button("Analyze"):
                with st.spinner("Processing your query..."):
                    try:
                        # First attempt with SmartDataframe
                        improved_prompt = improve_prompt(llm, user_input, df.columns.tolist())
                        answer = smart_df.chat(improved_prompt)

                        if is_visualization(answer):
                            if os.path.exists(image_path):
                                st.image(image_path, caption="Generated Visualization")
                            else:
                                st.warning("Visualization generated but could not be displayed.")

                        elif is_meaningful_text(answer):
                            st.write("**Text Output:**", answer)

                        else:
                            st.warning("Retry failed. Using fallback response...")
                            fallback_response = chatbot_fallback(llm, user_input, df.columns.tolist())
                            st.write("**Fallback Response:**", fallback_response)
                            if not is_meaningful_text(fallback_response):
                                st.warning("Fallback not meaningful. Chatting directly with LLM...")
                                llm_response = llm_direct_chat(llm, user_input)
                                st.write("**LLM Response:**", llm_response)

                    except NoCodeFoundError as e:
                        st.warning(f"PandasAI error: {e}. Improving prompt...")
                        improved_prompt = improve_prompt(llm, user_input, df.columns.tolist())
                        st.write("**Improved Prompt:**", improved_prompt)
                        
                        try:
                            # Retry with improved prompt
                            retry_answer = smart_df.chat(improved_prompt)
                            if is_visualization(retry_answer):
                                if os.path.exists(image_path):
                                    st.image(image_path, caption="Retry Visualization")
                                else:
                                    st.warning("Retry visualization generated but could not be displayed.")

                            elif is_meaningful_text(retry_answer):
                                st.write("**Retry Text Output:**", retry_answer)

                            else:
                                st.warning("Retry failed. Using fallback response...")
                                fallback_response = chatbot_fallback(llm, user_input, df.columns.tolist())
                                st.write("**Fallback Response:**", fallback_response)
                                if not is_meaningful_text(fallback_response):
                                    st.warning("Fallback not meaningful. Chatting directly with LLM...")
                                    llm_response = llm_direct_chat(llm, user_input)
                                    st.write("**LLM Response:**", llm_response)

                        except Exception as retry_e:
                            st.warning(f"Retry error: {retry_e}. Using fallback response...")
                            fallback_response = chatbot_fallback(llm, user_input, df.columns.tolist())
                            st.write("**Fallback Response:**", fallback_response)

                            if not is_meaningful_text(fallback_response):
                                st.warning("Fallback not meaningful. Chatting directly with LLM...")
                                llm_response = llm_direct_chat(llm, user_input)
                                st.write("**LLM Response:**", llm_response)

                    except Exception as e:
                        st.warning(f"Error: {e}. Using fallback response...")
                        fallback_response = chatbot_fallback(llm, user_input, df.columns.tolist())
                        st.write("**Fallback Response:**", fallback_response)
                        if not is_meaningful_text(fallback_response):
                            st.warning("Fallback not meaningful. Chatting directly with LLM...")
                            llm_response = llm_direct_chat(llm, user_input)
                            st.write("**LLM Response:**", llm_response)

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("Upload a CSV, Excel, or SQLite database file and enter a prompt to proceed.")

if __name__ == "__main__":
    main()