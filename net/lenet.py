import torch
import torch.nn as nn
import torch.nn.functional as F


# class LeNet(nn.Module):
#   def __init__(self):
#     super(LeNet, self).__init__()
#     self.conv1 = nn.Conv2d(1, 20, kernel_size=5, bias=False)
#     self.conv2 = nn.Conv2d(20, 50, kernel_size=5, bias=False)
#     self.fc1 = nn.Linear(800, 500)
#     self.fc2 = nn.Linear(500, 6)
#
#   def forward(self, x):
#     out = self.conv1(x)
#     out = F.relu(F.max_pool2d(out, 2))
#     out = self.conv2(out)
#     out = F.relu(F.max_pool2d(out, 2))
#     out = out.view(-1, 800)
#     out = self.fc1(out)
#     out = F.relu(out)
#     out = self.fc2(out)
#     return out
class LeNet(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = torch.nn.Sequential(
      torch.nn.Conv2d(1, 20, kernel_size=5),
      torch.nn.MaxPool2d(2),
      torch.nn.BatchNorm2d(20),
      torch.nn.ReLU(),
      torch.nn.Conv2d(20, 50, kernel_size=5),
      torch.nn.MaxPool2d(2),
      torch.nn.BatchNorm2d(50),
      torch.nn.ReLU()
    )

    self.dense = torch.nn.Sequential(
      torch.nn.Linear(800, 500),
      torch.nn.BatchNorm1d(500),
      torch.nn.ReLU(),
      torch.nn.Linear(500, 10)
    )

  def forward(self, x):
    x = self.conv(x)
    x = torch.flatten(x, start_dim=1)
    x = self.dense(x)
    return x

if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)


  net = LeNet()
  for m in net.modules():
    m.register_forward_hook(hook)
  y = net(torch.randn(1, 1, 28, 28))
  print(y.size())
