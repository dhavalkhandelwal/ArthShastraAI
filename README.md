ArthShastraAI 💸
An advanced, multimodal financial assistant that leverages Generative AI for deep portfolio analysis, risk assessment, and intelligent Q&A. This tool empowers users to upload their portfolio data, receive comprehensive quantitative analysis, and get AI-driven advice for optimization.

🚀 Core Features
Portfolio Optimizer Pro: Upload a CSV of historical asset prices to receive an in-depth analysis, including:

Automated data frequency detection (daily, weekly, monthly).

Comprehensive statistical summary tables.

Rich visualizations including cumulative returns, risk-return scatter plots, correlation heatmaps, and rolling volatility charts.

Calculation and visualization of the Efficient Frontier with optimal portfolio compositions (Max Sharpe, Global Minimum Volatility).

Intelligent Q&A Assistant: An AI-powered chatbot that can answer financial questions. It leverages a Retrieval-Augmented Generation (RAG) system to provide answers based on both its general knowledge and information from documents you upload.

Persistent Knowledge Base: Upload your own text documents (.txt, .md) to create a custom, persistent knowledge base. The AI assistant will use this information to provide more tailored answers.

🧠 Key Concepts Implemented
This project stands at the intersection of quantitative finance and cutting-edge artificial intelligence.

Financial Engineering & Quantitative Analysis
Modern Portfolio Theory (MPT): Core concepts are used to calculate and visualize the Efficient Frontier, allowing for the identification of optimal portfolios.

Risk & Return Metrics: Calculation of key performance indicators, including:

Sharpe Ratio: For risk-adjusted return.

Max Drawdown: To quantify potential downside risk.

Skewness & Kurtosis: To understand the shape and tail risk of the return distribution.

Value at Risk (VaR): Including the Cornish-Fisher VaR, which accounts for non-normal distributions.

Conditional Value at Risk (CVaR): To measure the expected shortfall in the tail of the distribution.

Time Series Analysis: Calculation of rolling volatility and cumulative returns to understand asset behavior over time.

Correlation Analysis: Generation of correlation matrices to assess diversification within the portfolio.

AI & NLP Concepts
Large Language Models (LLMs): Utilizes Google's Gemini model via LangChain for generating human-like analysis, summaries, and recommendations.

Retrieval-Augmented Generation (RAG): The Q&A system is built on a RAG architecture, which grounds the LLM's responses in specific, user-provided documents to reduce hallucinations and increase accuracy.

Sentence Embeddings: Employs advanced Transformer models (all-MiniLM-L6-v2) to convert text into meaningful vector representations for semantic understanding.

Tokenization: While handled implicitly by the embedding model, tokenization is a fundamental first step in the NLP pipeline used here to process all text inputs.

Vector Database: Uses FAISS (Facebook AI Similarity Search) as a high-speed, local vector store to index document embeddings and perform efficient similarity searches for the RAG system.

Prompt Engineering: Carefully crafted prompts are used to instruct the LLM on how to analyze financial data and structure its advice professionally.

🛠️ Technology Stack
Frontend: Streamlit

AI/LLM Framework: LangChain

Generative AI Model: Google Gemini

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

NLP Embeddings: Sentence-Transformers (via HuggingFace)

Vector Store: FAISS

Financial Analysis: SciPy

⚙️ Setup and Installation
Follow these steps to get the application running locally.

Prerequisites
Python 3.10 or higher

Git

1. Clone the Repository
Bash

git clone https://github.com/dhavalkhandelwal/ArthShastraAI.git
cd ArthShastraAI
2. Create and Activate a Virtual Environment
On Windows:

Bash

python -m venv venv
.\venv\Scripts\activate
On macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Set Up Your API Key
Open the app.py file in a code editor.

Find the following line:

Python

GEMINI_API_KEY = "your_api_key_here"
Replace "your_api_key_here" with your actual Google Gemini API key.


5. Run the Application
Bash

streamlit run app.py
The application should now be open and running in your web browser!

📖 Usage
Portfolio Optimizer Pro: Navigate to this mode and upload the provided comprehensive_portfolio.csv or your own CSV file containing historical price data. The app will automatically guide you through the analysis.

Upload Knowledge (RAG): Go to this mode to upload .txt or .md files. This will add them to the AI's knowledge base.

Ask Anything (Q&A): Use this chat interface to ask general financial questions or questions specifically about the documents you've uploaded.

📂 Project Structure
ArthShastraAI/
├── data/                  # Local storage for session data
│   ├── faiss_index/       # FAISS vector store for RAG
│   └── chat_history.json  # Persistent chat logs
├── finance_toolkit.py     # Module for all financial calculations
├── app.py                 # Main Streamlit application file
└── requirements.txt       # Python dependencies
