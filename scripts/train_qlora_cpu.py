#!/usr/bin/env python3
"""
AHM Finance - Loan Officer Assistant (QLoRA Fine-Tuning for Windows - CPU Version)

This script fine-tunes a small LLaMA-style model with QLoRA using CPU training
to completely avoid Triton and bitsandbytes dependency issues on Windows.

Designed for Windows systems without GPU dependencies.
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import json

# Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for CPU training
MAX_SEQ_LEN = 512  # Reduced for CPU training
BATCH_SIZE = 1  # Small batch size for CPU
GR_ACCUM = 4
LEARNING_RATE = 1e-4  # Lower learning rate for CPU
NUM_EPOCHS = 2  # Fewer epochs for faster training
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
    
    # Load base model (CPU version - no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",  # Force CPU
    )
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=8,  # Reduced rank for CPU
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
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )
    
    # Add labels for training (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def train_model(model, tokenizer, dataset):
    """Train the model using QLoRA."""
    print("Starting training on CPU...")
    print("Note: CPU training will be slower than GPU training.")
    
    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Prepare training arguments (CPU optimized)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=False,  # Disable fp16 for CPU
        logging_steps=5,
        save_steps=50,
        warmup_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        dataloader_pin_memory=False,  # Windows compatibility
        no_cuda=True,  # Force CPU
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Training started. This may take a while on CPU...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed successfully!")

def main():
    """Main training function."""
    print("=== AHM Finance Loan Officer Model Training (CPU Version) ===")
    print("This version uses CPU training to avoid GPU dependency issues.")
    print("Training will be slower but more compatible with Windows systems.")
    print()
    
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
