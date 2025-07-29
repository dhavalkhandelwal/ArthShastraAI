# 💸 ArthShastraAI – Multimodal Financial Assistant

**ArthShastraAI** is an advanced, multimodal financial assistant built to empower investors and learners with AI-driven portfolio optimization, risk analysis, and intelligent financial Q&A. This project was developed as a personal innovation to explore the intersection of **quantitative finance** and **Generative AI**.

> 🔬 Built for deep financial insight. Powered by AI.

---

## 🚀 Core Features

### 📈 Portfolio Optimizer Pro

Upload a CSV of historical asset prices and get:

- 📊 Automated data frequency detection (daily/weekly/monthly)
- 📋 Statistical summary tables
- 📉 Rich visualizations:
  - Cumulative Returns
  - Risk-Return Scatter Plots
  - Correlation Heatmaps
  - Rolling Volatility Charts
- 📌 Efficient Frontier with:
  - Maximum Sharpe Ratio Portfolio
  - Global Minimum Volatility Portfolio

### 🤖 Intelligent Q&A Assistant (RAG-powered)

- Ask general financial questions or portfolio-specific queries
- Upload `.txt` or `.md` files for context-aware answers
- Backed by a **Retrieval-Augmented Generation (RAG)** system

### 📚 Persistent Knowledge Base

- Documents you upload stay in a custom knowledge base
- The assistant uses them for smarter, tailored responses

---

## 🧠 Key Concepts Implemented

### 🔢 Financial Engineering & Quantitative Analysis

- **Modern Portfolio Theory (MPT)** – Efficient Frontier
- **Risk & Return Metrics**:
  - Sharpe Ratio
  - Max Drawdown
  - Skewness & Kurtosis
  - Value at Risk (VaR) – includes Cornish-Fisher extension
  - Conditional VaR (CVaR)
- **Time Series Analysis** – Rolling volatility, Cumulative returns
- **Correlation Analysis** – Heatmaps & diversification insights

### 🧬 AI & NLP Concepts

- **Large Language Models** – Google Gemini via LangChain
- **RAG (Retrieval-Augmented Generation)** architecture
- **Sentence Embeddings** – `all-MiniLM-L6-v2` from HuggingFace
- **Vector Store** – FAISS for fast, local similarity search
- **Prompt Engineering** – Financial-specific prompting

---

## 🛠️ Tech Stack

| Layer         | Technologies                            |
|---------------|------------------------------------------|
| Frontend      | Streamlit                                |
| AI/NLP        | LangChain, Gemini, SentenceTransformers  |
| Data Analysis | Pandas, NumPy, SciPy                     |
| Visuals       | Matplotlib, Seaborn                      |
| Storage       | FAISS (vector database)                  |

---

## ⚙️ Setup and Installation

### ✅ Prerequisites

- Python 3.10+
- Git

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/dhavalkhandelwal/ArthShastraAI.git
cd ArthShastraAI
