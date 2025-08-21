# AHM Finance Loan Officer Assistant

A comprehensive AI-powered loan application analysis system with dual model support - combining the speed of Ollama with the specialization of fine-tuned models for optimal loan officer decision-making.

## Prerequisites

- Python 3.8 or higher
- Ollama (for LLM functionality)
- Command Prompt (cmd) - **Use cmd, not PowerShell**
- NVIDIA GPU with 4GB+ VRAM (for training)

## Quick Start (Command Prompt)

### Option 1: Using Batch Files (Recommended)

1. **Open Command Prompt (cmd)** - Right-click in the project folder and select "Open command window here" or navigate to the project directory in cmd.

2. **Run Setup** (choose one):
   ```cmd
   # For web app only (faster setup)
   setup.bat
   
   # For training + web app (Windows compatible)
   setup_windows.bat
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

- 🤖 **Dual AI Models**: Choose between Ollama (fast) and fine-tuned model (specialized)
- 📊 **File Upload**: Analyze Excel/CSV files with loan application data
- 📝 **Manual Input**: Enter loan details manually for quick analysis
- 🎯 **Specialized Analysis**: Fine-tuned model trained specifically for loan officer tasks
- ⚡ **Fast Processing**: Ollama integration for quick general analysis
- 📈 **Comprehensive Reports**: Approval recommendations, risk assessment, and documentation requirements

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
│   ├── app.py              # Main Streamlit application (integrated)
│   ├── train_qlora_cpu.py  # QLoRA fine-tuning script (CPU)
│   └── inference_cpu.py    # Model inference script (CPU)
├── setup.bat               # Setup script for cmd
├── run_app.bat             # Application launcher for cmd
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## AI Models

### 🤖 Ollama Model (Gemma 2B)
- **Purpose**: General-purpose financial analysis
- **Speed**: Fast processing
- **Requirements**: Ollama must be running (`ollama serve`)
- **Best for**: Quick analysis and general financial insights

### 🎯 Fine-tuned Model (TinyLlama 1.1B)
- **Purpose**: Specialized loan officer analysis
- **Training**: Pre-trained on loan application data
- **Features**: Approval recommendations, risk assessment, documentation requirements
- **Best for**: Detailed loan analysis and decision support

## Training (Optional)

The fine-tuned model is already trained and ready to use. If you want to retrain:

```cmd
python scripts/train_qlora_cpu.py
```

This will:
- Load the sample dataset
- Fine-tune TinyLlama 1.1B using QLoRA (CPU version)
- Save the model to `./models/finetuned-model`
- **Windows compatible** - no GPU dependencies

## Development

To modify the application:

1. Edit `scripts/app.py` for the main application logic
2. Edit `scripts/train_qlora_cpu.py` for training modifications
3. Edit `scripts/inference_cpu.py` for inference modifications
4. Update `requirements.txt` for new dependencies
5. Test changes by running `run_app.bat`

## Notes

- **Always use Command Prompt (cmd) for this project**
- The application requires an internet connection for the first run to download the LLM model
- Large datasets may take longer to process
