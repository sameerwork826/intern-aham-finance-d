#!/usr/bin/env python3
"""
AHM Finance - Loan Officer Assistant (Inference - CPU Version)

This script uses the fine-tuned model to analyze loan applications and provide
recommendations for loan officers. CPU version without GPU dependencies.
"""

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Configuration
MODEL_PATH = "./models/finetuned-model"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    """Load the fine-tuned model and tokenizer."""
    print("Loading fine-tuned model...")
    
    # Load base model and tokenizer (CPU version)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",  # Force CPU
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    return model, tokenizer

def create_prompt(age, income, loan_amount, credit_score, employment_status="Full-time", anomalies="None"):
    """Create a prompt for loan analysis."""
    prompt = f"""
    Loan Application Analysis:
    - Age: {age}
    - Income: ${income}
    - Loan Amount: ${loan_amount}
    - Credit Score: {credit_score}
    - Employment Status: {employment_status}
    - Anomalies: {anomalies}
    
    Please analyze this loan application and provide:
    1. Approval recommendation
    2. Risk assessment
    3. Required documentation
    4. Reasoning for decision
    """
    return prompt.strip()

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response using the fine-tuned model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    response = response[len(prompt):].strip()
    
    return response

def analyze_loan_application(age, income, loan_amount, credit_score, employment_status="Full-time", anomalies="None"):
    """Analyze a loan application using the fine-tuned model."""
    print("=== Loan Application Analysis ===")
    print(f"Age: {age}")
    print(f"Income: ${income}")
    print(f"Loan Amount: ${loan_amount}")
    print(f"Credit Score: {credit_score}")
    print(f"Employment Status: {employment_status}")
    print(f"Anomalies: {anomalies}")
    print()
    
    # Load model
    model, tokenizer = load_model()
    
    # Create prompt
    prompt = create_prompt(age, income, loan_amount, credit_score, employment_status, anomalies)
    
    # Generate response
    print("Generating analysis...")
    response = generate_response(model, tokenizer, prompt)
    
    print("=== Analysis Results ===")
    print(response)
    
    return response

def analyze_from_csv(csv_path, row_index=None):
    """Analyze loan applications from a CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if row_index is not None:
        # Analyze specific row
        if row_index < len(df):
            row = df.iloc[row_index]
            analyze_loan_application(
                age=row.get('age', 'N/A'),
                income=row.get('income', 'N/A'),
                loan_amount=row.get('loan_amount', 'N/A'),
                credit_score=row.get('credit_score', 'N/A'),
                employment_status=row.get('employment_status', 'Full-time'),
                anomalies=row.get('anomalies', 'None')
            )
        else:
            print(f"Row {row_index} not found. CSV has {len(df)} rows.")
    else:
        # Analyze first few rows
        print(f"Analyzing first 3 rows from {len(df)} total rows...")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\n{'='*50}")
            print(f"ROW {i+1}")
            print(f"{'='*50}")
            analyze_loan_application(
                age=row.get('age', 'N/A'),
                income=row.get('income', 'N/A'),
                loan_amount=row.get('loan_amount', 'N/A'),
                credit_score=row.get('credit_score', 'N/A'),
                employment_status=row.get('employment_status', 'Full-time'),
                anomalies=row.get('anomalies', 'None')
            )

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="AHM Finance Loan Officer Assistant (CPU Version)")
    parser.add_argument("--age", type=int, help="Applicant age")
    parser.add_argument("--income", type=int, help="Annual income")
    parser.add_argument("--loan_amount", type=int, help="Requested loan amount")
    parser.add_argument("--credit_score", type=int, help="Credit score")
    parser.add_argument("--employment_status", type=str, default="Full-time", help="Employment status")
    parser.add_argument("--anomalies", type=str, default="None", help="Any anomalies in application")
    parser.add_argument("--csv", type=str, help="Path to CSV file with loan data")
    parser.add_argument("--row", type=int, help="Row index to analyze from CSV (0-based)")
    
    args = parser.parse_args()
    
    if args.csv:
        # Analyze from CSV file
        analyze_from_csv(args.csv, args.row)
    elif all([args.age, args.income, args.loan_amount, args.credit_score]):
        # Analyze individual application
        analyze_loan_application(
            age=args.age,
            income=args.income,
            loan_amount=args.loan_amount,
            credit_score=args.credit_score,
            employment_status=args.employment_status,
            anomalies=args.anomalies
        )
    else:
        print("Please provide either individual parameters or a CSV file.")
        print("\nExamples:")
        print("Individual application:")
        print("  python inference_cpu.py --age 29 --income 32000 --loan_amount 12000 --credit_score 600")
        print("\nFrom CSV file:")
        print("  python inference_cpu.py --csv dataset/sample_loan_dataset.csv --row 1")

if __name__ == "__main__":
    main()
