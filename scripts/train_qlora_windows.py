#!/usr/bin/env python3
"""
AHM Finance - Loan Officer Assistant (QLoRA Fine-Tuning for Windows)

This script fine-tunes a small LLaMA-style model with QLoRA using standard PEFT
(without Unsloth) to avoid Triton dependency issues on Windows.

Designed for NVIDIA GTX 1650 (~4GB VRAM) with default base TinyLlama 1.1B.
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import json

# Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for 4GB VRAM
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1  # Reduced for Windows compatibility
GR_ACCUM = 8
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
        
        # Create the full text for training
        full_text = f"{prompt.strip()}\n\n{response.strip()}"
        
        training_data.append({
            "text": full_text
        })
    
    return Dataset.from_list(training_data)

def setup_model_and_tokenizer():
    """Setup the model and tokenizer for fine-tuning."""
    print("Setting up model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )

def train_model(model, tokenizer, dataset):
    """Train the model using QLoRA."""
    print("Starting training...")
    
    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Prepare training arguments
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
        dataloader_pin_memory=False,  # Windows compatibility
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
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
    print("=== AHM Finance Loan Officer Model Training (Windows Compatible) ===")
    
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
