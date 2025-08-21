#!/usr/bin/env python3
"""
AHM Finance - Loan Officer Assistant (QLoRA Fine-Tuning)

This script fine-tunes a small LLaMA-style model with QLoRA (via Unsloth + PEFT)
so loan officers can ask: "Why did the model reject this application, and what docs are needed?"

Designed for NVIDIA GTX 1650 (~4GB VRAM) with default base TinyLlama 1.1B.
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel
import pandas as pd
import json

# Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for 4GB VRAM
MAX_SEQ_LEN = 1024
BATCH_SIZE = 2
GR_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
OUTPUT_DIR = "./models/finetuned-model"

def load_and_prepare_data():
    """Load and prepare the loan dataset for training."""
    print("Loading dataset...")
    
    # Load the sample dataset
    df = pd.read_csv("dataset/sample_loan_dataset.csv")
    
    # Create training examples
    training_data = []
    
    for _, row in df.iterrows():
        # Create a prompt for loan analysis
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
        
        # Create response based on data
        if row.get('credit_score', 0) >= 650 and row.get('income', 0) > 30000:
            response = f"""
            APPROVAL RECOMMENDATION: APPROVED
            
            Risk Assessment: LOW to MEDIUM
            - Credit score of {row.get('credit_score')} is acceptable
            - Income of ${row.get('income')} provides adequate repayment capacity
            - Loan-to-income ratio is reasonable
            
            Required Documentation:
            - Proof of income (paystubs, tax returns)
            - Bank statements (3 months)
            - Credit report
            - Employment verification
            
            Reasoning: Applicant meets basic credit and income requirements for loan approval.
            """
        else:
            response = f"""
            APPROVAL RECOMMENDATION: DECLINED
            
            Risk Assessment: HIGH
            - Credit score of {row.get('credit_score')} is below minimum threshold
            - Income may be insufficient for loan amount
            - Additional risk factors present
            
            Required Documentation (for reconsideration):
            - Detailed explanation of credit issues
            - Additional income verification
            - Co-signer information
            - Collateral documentation
            
            Reasoning: Applicant does not meet minimum credit and income requirements.
            """
        
        training_data.append({
            "prompt": prompt.strip(),
            "response": response.strip(),
            "text": f"{prompt.strip()}\n\n{response.strip()}"
        })
    
    return Dataset.from_list(training_data)

def setup_model_and_tokenizer():
    """Setup the model and tokenizer for fine-tuning."""
    print("Setting up model and tokenizer...")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,  # Auto detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def train_model(model, tokenizer, dataset):
    """Train the model using QLoRA."""
    print("Starting training...")
    
    # Prepare training arguments
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
    )
    
    # Setup trainer
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=FastLanguageModel.get_train_data_collator(),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed successfully!")

def main():
    """Main training function."""
    print("=== AHM Finance Loan Officer Model Training ===")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    dataset = load_and_prepare_data()
    print(f"Loaded {len(dataset)} training examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Train the model
    train_model(model, tokenizer, dataset)
    
    print("=== Training Complete ===")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("You can now use the model with inference.py")

if __name__ == "__main__":
    main()
