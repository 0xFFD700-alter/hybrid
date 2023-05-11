import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from fedlab.utils.dataset.partition import CIFAR10Partitioner


def load_data(num_clients, data_dir):
    trainloaders = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(data_dir, train=True, download=True, transform=transform)
    validset = CIFAR10(data_dir, train=False, download=True, transform=transform)
    hetero_dir_part = CIFAR10Partitioner(trainset.targets, num_clients, balance=True, partition='dirichlet', dir_alpha=0.5, verbose=False)
    for client in range(num_clients):
        indices = hetero_dir_part.client_dict[client]
        trainloaders.append(
            DataLoader(trainset, batch_size=32, sampler=SubsetRandomSampler(indices))
        )
    validloader = DataLoader(validset, batch_size=32)
    return trainloaders, validloader


# def train(net, trainloader, epochs):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()

# def test(net, testloader):
#     """Validate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     return loss, accuracy


def main():
    trainloaders, testloader = load_data(32, "./data")
    for client in range(32):
        print(len(trainloaders[client]))
    print(len(testloader))

if __name__=='__main__':
    main()