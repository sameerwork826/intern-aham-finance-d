# AHM Finance Loan Officer Project

A Streamlit-based application for financial data analysis using LLM (Large Language Model) powered insights.

## Prerequisites

- Python 3.8 or higher
- Ollama (for LLM functionality)
- Command Prompt (cmd) - **Use cmd, not PowerShell**

## Quick Start (Command Prompt)

### Option 1: Using Batch Files (Recommended)

1. **Open Command Prompt (cmd)** - Right-click in the project folder and select "Open command window here" or navigate to the project directory in cmd.

2. **Run Setup**:
   ```cmd
   setup.bat
   ```

3. **Start the Application**:
   ```cmd
   run_app.bat
   ```

### Option 2: Manual Setup

1. **Open Command Prompt (cmd)** and navigate to the project directory:
   ```cmd
   cd C:\path\to\ahm_finance_loan_officer_project
   ```

2. **Install Python dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Install Ollama** (if not already installed):
   - Download from: https://ollama.ai/
   - Install and restart cmd

4. **Pull the required model**:
   ```cmd
   ollama pull gemma:2b
   ```

5. **Run the application**:
   ```cmd
   run_app.bat
   ```

## Usage

1. The application will open in your default browser at `http://localhost:8501`
2. Upload your financial Excel or CSV file
3. Ask questions about your data
4. Get AI-powered insights and analysis

## Features

- 📊 Financial data upload and preview
- 🤖 LLM-powered data analysis
- 💡 Interactive Q&A interface
- 📈 Support for Excel and CSV files

## Troubleshooting

### Common Issues

1. **"ollama not found" error**:
   - Make sure Ollama is installed from https://ollama.ai/
   - Restart Command Prompt after installation
   - Run `ollama --version` to verify installation

2. **Model not found error**:
   - Run `ollama pull gemma:2b` in cmd
   - Wait for the model to download completely

3. **PowerShell issues**:
   - **Always use Command Prompt (cmd), not PowerShell**
   - If you're in PowerShell, type `cmd` to switch to Command Prompt

### Getting Help

- Ensure you're using Command Prompt (cmd), not PowerShell
- Check that all dependencies are installed: `pip list`
- Verify Ollama is running: `ollama list`

## Project Structure

```
ahm_finance_loan_officer_project/
├── dataset/
│   └── sample_loan_dataset.csv
├── scripts/
│   ├── app.py              # Main Streamlit application (API version)
│   ├── train_qlora.py      # QLoRA fine-tuning script
│   └── inference.py        # Model inference script
├── setup.bat               # Setup script for cmd
├── run_app.bat             # Application launcher for cmd
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Training and Inference

### Training the Model

To train your own fine-tuned model:

```cmd
python scripts/train_qlora.py
```

This will:
- Load the sample dataset
- Fine-tune TinyLlama 1.1B using QLoRA
- Save the model to `./models/finetuned-model`

### Using the Trained Model

Analyze individual loan applications:

```cmd
python scripts/inference.py --age 29 --income 32000 --loan_amount 12000 --credit_score 600
```

Or analyze from CSV file:

```cmd
python scripts/inference.py --csv dataset/sample_loan_dataset.csv --row 1
```

## Development

To modify the application:

1. Edit `scripts/app.py` for the main application logic
2. Edit `scripts/train_qlora.py` for training modifications
3. Edit `scripts/inference.py` for inference modifications
4. Update `requirements.txt` for new dependencies
5. Test changes by running `run_app.bat`

## Notes

- **Always use Command Prompt (cmd) for this project**
- The application requires an internet connection for the first run to download the LLM model
- Large datasets may take longer to process
