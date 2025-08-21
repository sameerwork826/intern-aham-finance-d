import streamlit as st
import pandas as pd
import requests
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def get_ollama_analysis(prompt):
    """Get analysis from Ollama model."""
    try:
        # Prepare the request
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                return "❌ No response received from Ollama"
        else:
            return f"❌ Error: HTTP {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Make sure Ollama is running with: `ollama serve`"
    except requests.exceptions.Timeout:
        return "❌ Request timed out. The model is taking too long to respond."
    except Exception as e:
        return f"❌ Error: {str(e)}"

@st.cache_resource
def load_finetuned_model():
    """Load the fine-tuned model (cached)."""
    try:
        MODEL_PATH = "./models/finetuned-model"
        BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load fine-tuned model: {str(e)}")
        return None, None

def get_finetuned_analysis(prompt):
    """Get analysis from fine-tuned model."""
    try:
        model, tokenizer = load_finetuned_model()
        
        if model is None or tokenizer is None:
            return "❌ Fine-tuned model not available. Please ensure the model is trained and saved."
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response if response else "❌ No response generated from fine-tuned model"
        
    except Exception as e:
        return f"❌ Error with fine-tuned model: {str(e)}"

st.set_page_config(page_title="AHM Finance Loan Officer Assistant", layout="wide")

st.title("🏦 AHM Finance Loan Officer Assistant")
st.markdown("AI-powered loan application analysis with multiple model options")

# Sidebar for model selection
st.sidebar.header("🤖 Model Selection")
model_option = st.sidebar.selectbox(
    "Choose your AI model:",
    ["Ollama (Gemma 2B)", "Fine-tuned Model (TinyLlama)"],
    help="Ollama: Fast, general-purpose model. Fine-tuned: Specialized for loan analysis."
)

# Model configuration
if model_option == "Ollama (Gemma 2B)":
    st.sidebar.info("💡 **Ollama Model**: General-purpose AI for financial analysis")
    st.sidebar.markdown("**Requirements**: Ollama must be running (`ollama serve`)")
else:
    st.sidebar.info("🎯 **Fine-tuned Model**: Specialized for loan officer tasks")
    st.sidebar.markdown("**Features**: Trained specifically on loan data")

# Main content
st.header("📊 Loan Application Analysis")

# File upload
uploaded_file = st.file_uploader(
    "Upload loan application data (Excel/CSV)", 
    type=["xlsx", "csv"],
    help="Upload a file with loan application data to analyze"
)

# Manual input option
st.subheader("📝 Or Enter Application Details Manually")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    employment_status = st.selectbox("Employment Status", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    anomalies = st.text_input("Anomalies (optional)", placeholder="e.g., late payments, bankruptcy")

# Analysis button
if st.button("🔍 Analyze Loan Application", type="primary"):
    if uploaded_file is not None:
        # Analyze uploaded file
        st.subheader("📂 File Analysis")
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.dataframe(df.head())
        
        # Analyze first few rows
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            st.markdown(f"### Row {i+1} Analysis")
            
            # Create analysis prompt
            prompt = f"""
            Loan Application Analysis:
            - Age: {row.get('age', 'N/A')}
            - Income: ${row.get('income', 'N/A')}
            - Loan Amount: ${row.get('loan_amount', 'N/A')}
            - Credit Score: {row.get('credit_score', 'N/A')}
            - Employment Status: {row.get('employment_status', 'N/A')}
            - Anomalies: {row.get('anomalies', 'None')}
            
            Please analyze this loan application and provide:
            1. Approval recommendation
            2. Risk assessment
            3. Required documentation
            4. Reasoning for decision
            """
            
            # Get analysis based on selected model
            with st.spinner(f"Analyzing row {i+1}..."):
                if model_option == "Ollama (Gemma 2B)":
                    analysis = get_ollama_analysis(prompt)
                else:
                    analysis = get_finetuned_analysis(prompt)
            
            st.markdown(analysis)
            st.divider()
    
    else:
        # Analyze manual input
        st.subheader("📋 Manual Input Analysis")
        
        # Create analysis prompt
        prompt = f"""
        Loan Application Analysis:
        - Age: {age}
        - Income: ${income}
        - Loan Amount: ${loan_amount}
        - Credit Score: {credit_score}
        - Employment Status: {employment_status}
        - Anomalies: {anomalies if anomalies else 'None'}
        
        Please analyze this loan application and provide:
        1. Approval recommendation
        2. Risk assessment
        3. Required documentation
        4. Reasoning for decision
        """
        
        # Get analysis based on selected model
        with st.spinner("Analyzing loan application..."):
            if model_option == "Ollama (Gemma 2B)":
                analysis = get_ollama_analysis(prompt)
            else:
                analysis = get_finetuned_analysis(prompt)
        
        st.markdown("### Analysis Results")
        st.markdown(analysis)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏦 <strong>AHM Finance Loan Officer Assistant</strong> | 
    Powered by AI for smarter loan decisions</p>
</div>
""", unsafe_allow_html=True)
