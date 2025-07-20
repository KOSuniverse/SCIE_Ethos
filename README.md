# Jarvis Knowledge Base Assistant

A modular, AI-powered knowledge base assistant that integrates with Google Drive to provide intelligent document analysis, Q&A capabilities, and predictive modeling.

## Project Structure

```
├── main.py                     # Main Streamlit application
├── llm_client.py              # OpenAI integration and LLM operations
├── modeling.py                # Predictive modeling functionality
├── gdrive_utils.py            # Google Drive service account integration
├── requirements.txt           # Python dependencies
├── utils/
│   ├── __init__.py           # Package initialization
│   ├── retry.py              # OpenAI retry logic
│   ├── gdrive.py             # Google Drive helper functions
│   ├── metadata.py           # Metadata operations
│   ├── text_utils.py         # Text processing and chunking
│   ├── excel_qa.py           # Excel analysis with charts
│   └── column_mapping.py     # Column mapping and aliases
└── README.md                 # This file
```

## Features

- 🔍 **Intelligent Document Search** - Hybrid keyword + semantic search
- 📊 **Excel Q&A with Charts** - Automated data analysis and visualization
- 🤖 **AI-Powered Insights** - Root cause analysis and business insights
- 📈 **Predictive Modeling** - Build, train, and deploy ML models
- 💾 **Answer Caching** - Learn from previous queries
- 🔧 **Column Mapping** - Standardize Excel column names
- 📁 **Google Drive Integration** - Seamless file access and storage

## Setup

### 🌐 **Cloud-Native Deployment (Recommended):**

**No local installation needed!** This runs entirely in the cloud:

1. **Upload Code to GitHub** (through GitHub web interface or GitHub Desktop):
   - Create a new repository on GitHub.com
   - Upload all your `.py` files, `requirements.txt`, and `README.md`
   - **Your documents stay in Google Drive - don't upload them to GitHub!**

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `main.py`
   - Streamlit Cloud automatically installs all dependencies from `requirements.txt`

3. **Configure Secrets** in Streamlit Cloud dashboard:
   - Add your `OPENAI_API_KEY`
   - Add your `[gdrive_service_account]` credentials (you already have these)

**That's it! Everything runs in the cloud.** ☁️

### 📁 **What Goes Where:**

**GitHub (Code Only):**
- ✅ `main.py`, `llm_client.py`, `modeling.py`, etc.
- ✅ `utils/` folder with all `.py` files
- ✅ `requirements.txt`, `README.md`, `.gitignore`
- ❌ **NO documents** (Excel, Word, PDF files)
- ❌ **NO secrets** (handled in Streamlit Cloud)

**Google Drive (Documents Only):**
- ✅ Your Excel files, Word docs, PDFs
- ✅ Metadata files (auto-generated)
- ✅ ML models (auto-saved)
- ❌ **NO code files**

### 💻 **No Command Line Needed:**

You can do everything through web interfaces:
- **GitHub.com** - Upload your code files using drag & drop
- **share.streamlit.io** - Deploy with a few clicks
- **Google Drive** - Your documents are already there!

*The git commands in the previous section are only if you prefer command line, but you can use GitHub's web interface instead.*

## Google Drive Structure Expected

```
Project_Root/
├── 01_Project_Plan/
│   └── _metadata/           # Metadata storage
├── 04_Data/
│   └── Models/             # ML model storage
└── [Other folders with documents]
```

## Usage

1. **Ask Questions** - Type natural language questions about your documents
2. **Edit Aliases** - Use the checkbox to standardize column names
3. **Generate Charts** - Excel files automatically offer chart generation
4. **Build Models** - Use AI to create predictive models from your data
5. **View History** - Check previous queries and results

## Dependencies

- **Streamlit** - Web interface and cloud deployment
- **OpenAI** - LLM capabilities and embeddings
- **Google Drive API** - File access using service account
- **Data Science Stack** - pandas, matplotlib, seaborn, scikit-learn
- **Document Processing** - openpyxl, python-docx, pdfplumber
- **ML Libraries** - scikit-learn, xgboost for predictive modeling

## Streamlit Cloud Deployment

This project is **100% cloud-native** and requires:

**✅ What You Need:**
- GitHub repository (free)
- Streamlit Cloud account (free) 
- Google Drive with your documents
- OpenAI API key
- Google service account (you have this)

**❌ What You DON'T Need:**
- Local Python installation
- Local file storage
- Local authentication files
- Local servers or databases

**🚀 Deployment Process:**
1. Push code to GitHub → 2. Connect to Streamlit Cloud → 3. Add secrets → 4. LIVE!

All processing happens in Streamlit's cloud infrastructure, accessing your Google Drive files through the service account.

## Note

The application uses service account credentials stored in Streamlit secrets for secure Google Drive access. No local authentication files needed.
