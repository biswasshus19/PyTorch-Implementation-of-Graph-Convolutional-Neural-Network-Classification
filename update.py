import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch_geometric.transforms import NormalizeFeatures

# Load and normalize dataset (Cora, you can switch to CiteSeer or PubMed)
dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]


# Define the Improved GAT model with 3 layers and residual connections
class ImprovedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(ImprovedGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.gat3 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)


# Hyperparameters
hidden_channels = 32
lr = 0.002
weight_decay = 1e-4
dropout_rate = 0.6
epochs = 200
early_stop_patience = 20

# Initialize model and optimizer
model = ImprovedGAT(
    in_channels=dataset.num_node_features,
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# Training function with early stopping
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluation function
@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]


# Early stopping and training loop
def main():
    best_val_acc = 0
    best_test_acc = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, val_acc, test_acc = evaluate()

        # Check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stop if validation accuracy doesn't improve for `early_stop_patience` epochs
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
              f"Test: {test_acc:.4f} (Best Test: {best_test_acc:.4f})")


if __name__ == "__main__":
    main()
