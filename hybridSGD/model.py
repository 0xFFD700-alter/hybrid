import torch
from torch import nn
from torch.nn import functional as F

class FedAvgCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# batch_size = 10

# class FedAvgCNN(nn.Module):
#     def __init__(self):
#         super(FedAvgCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
#         self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(18432, 128)
#         self.fc = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc(x)
#         output = F.log_softmax(x, dim=1)
#         return output

def main():
    imgs = torch.zeros((32, 3, 32, 32))
    net = FedAvgCNN(3, 10)
    logits = net(imgs)
    print(logits.shape)
    count = sum([param.nelement() for param in net.parameters()])
    print(count)

if __name__ == '__main__':
    main()
