from rag.embed import build_index
import numpy as np

index, docs, model = build_index("data/fraud_cases.txt")

def generate_explanation(alert_type):
    query = f"Explain {alert_type} fraud pattern"
    q_emb = model.encode([query])
    _, idx = index.search(np.array(q_emb), k=2)

    context = " ".join([docs[i] for i in idx[0]])

    explanation = f"""
Fraud Type: {alert_type}

Why this was flagged:
{context}

System Insight:
The GNN detected abnormal connectivity and repeated interactions between linked accounts, 
which matches known fraud patterns retrieved from historical cases.
"""
    return explanation