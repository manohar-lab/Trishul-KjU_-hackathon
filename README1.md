# TRISHUL – Graph-Based Fraud Detection System

TRISHUL is a prototype fraud detection platform that combines **Graph Neural Networks (GNNs)** with **Retrieval-Augmented Generation (RAG)** to detect and explain coordinated micro-fraud in financial transactions.

## 🚀 Key Features
- **Graph-Based Fraud Detection**
  - Models transactions as graphs (accounts as nodes, transactions as edges)
  - Detects coordinated multi-account fraud patterns using GNNs

- **Contextual Fraud Explanations (RAG)**
  - Enriches fraud alerts with historical fraud knowledge
  - Reduces false positives and improves analyst trust

- **Live Dashboard**
  - Auto-refreshing frontend
  - Displays transactions and fraud alerts in near real-time

## 🧠 Architecture Overview
1. Transaction ingestion (simulated real-time)
2. Graph construction from transaction data
3. GNN-based fraud prediction
4. RAG-based contextual explanation
5. Alert visualization on dashboard

## 🛠 Tech Stack
- **Backend:** Python, FastAPI
- **Machine Learning:** PyTorch, PyTorch Geometric
- **Graph Processing:** NetworkX
- **RAG:** Sentence Transformers, FAISS
- **Frontend:** HTML, CSS, JavaScript
- **Version Control:** Git, GitHub

## ▶️ How to Run Locally

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
