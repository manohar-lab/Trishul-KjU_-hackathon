from fastapi import FastAPI
from gnn.graph_builder import build_graph
from gnn.model import FraudGNN
from rag.retrieve import generate_explanation
import torch
import torch.nn.functional as F
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TRISHUL – Graph Fraud Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/transactions.csv")
data = build_graph("data/transactions.csv")

# ---------------------------
# Train GNN once at startup
# ---------------------------
model = FraudGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for _ in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

model.eval()

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "TRISHUL backend running"}

@app.get("/transactions")
def get_transactions():
    return df.to_dict(orient="records")

@app.get("/alerts")
def get_alerts():
    with torch.no_grad():
        out = model(data)
        preds = out.argmax(dim=1).tolist()

    alerts = []
    nodes = list(data.y.numpy().tolist())

    accounts = list(dict.fromkeys(list(df["src"]) + list(df["dst"])))

    for acc, pred in zip(accounts, preds):
        if pred == 1:
            alerts.append({
                "account": acc,
                "risk": "HIGH",
                "explanation": generate_explanation("micro-transaction ring")
            })

    return alerts