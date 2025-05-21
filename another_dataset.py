from torch_geometric.datasets import Planetoid
data = Planetoid(root='./data', name='PubMed')[0]
print(data)

##We use the GCNConv package in torch to build our GCN network.
import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.optim import Adam
class GCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, class_n):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, class_n)
  def forward(self, x, edge_index):
    x = torch.relu(self.conv1(x, edge_index))
    x = torch.dropout(x, p=0.5, train=self.training)
    x = self.conv2(x, edge_index)
    return torch.log_softmax(x, dim=1)
#Build the model and optimizer
model = GCN(data.num_features, 16, data.y.unique().shape[0])
opt = Adam(model.parameters(), 0.01, weight_decay=5e-4)
def train(its):
  model.train()
  for i in range(its):
    y = model(data.x, data.edge_index)
    loss = F.nll_loss(y[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()
    opt.zero_grad()
def test():
  model.eval()
  y = model(data.x, data.edge_index)
  right_n = torch.argmax(y[data.test_mask], 1) == data.y[data.test_mask]
  acc = right_n.sum()/data.test_mask.sum()
  print("Acc: ", acc)
def main():
  for i in range(10):
    train(1)
    test()
if __name__ == '__main__':
    main()
