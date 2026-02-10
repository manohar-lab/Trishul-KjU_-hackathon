import torch
import torch.nn.functional as F
from graph_builder import build_graph
from model import FraudGNN

data = build_graph("data/transactions.csv")
model = FraudGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
pred = model(data).argmax(dim=1)
print("\nNode Fraud Predictions:")
print(pred.tolist())