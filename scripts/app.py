import streamlit as st
import pandas as pd
import requests
import json
import time

st.set_page_config(page_title="Finance Intern Project", layout="wide")

st.title("📊 Finance Intern Project - LLM Powered Insights (API Version)")

# Upload Excel
uploaded_file = st.file_uploader("Upload your Finance Excel file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("📂 Data Preview")
    st.dataframe(df.head())

    # Ask user query
    query = st.text_area("Ask your question about this data:")
    
    if st.button("Generate Answer"):
        # Convert DataFrame to JSON (sampled to avoid overload)
        data_sample = df.head(50).to_json(orient="records")

        prompt = f"""
        You are a financial data analyst intern.
        Here is sample company financial data: {data_sample}

        Question: {query}

        Answer in clear bullet points.
        """

        # Use Ollama REST API instead of subprocess
        try:
            # Prepare the request
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False
            }
            
            # Show loading message
            with st.spinner("Generating response..."):
                response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    st.subheader("🤖 LLM Response")
                    st.write(result['response'])
                else:
                    st.error("No response received from Ollama")
            else:
                st.error(f"Error: HTTP {response.status_code}")
                st.info("Make sure Ollama is running. Start it with 'ollama serve' in cmd.")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Make sure Ollama is running.")
            st.info("Start Ollama with: ollama serve")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The model is taking too long to respond.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please ensure Ollama is installed and running. Visit https://ollama.ai/ for installation instructions.")
