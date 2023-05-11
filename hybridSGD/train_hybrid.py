import copy
from model import FedAvgCNN as net
from util import *
from dataset import load_data
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_clients = 32
num_clusters = 4
num_per_cluster = num_clients // num_clusters
in_channels = 3
num_classes = 10
data_dir = './data'
lr = 1e-3
loss_fn = torch.nn.CrossEntropyLoss().to(device)
save_accuracy_interval = 1
save_model_interval = 10
epochs = 200
local_epochs = 20


class Link(object):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def pass_link(self, pay_load):
        return pay_load


class FedAvgServer:
    def __init__(self, global_parameters, down_link):
        self.global_parameters = global_parameters
        self.down_link = down_link

    def download(self, user_idx):
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(self.down_link.pass_link(copy.deepcopy(self.global_parameters)))
        return local_parameters

    def upload(self, local_parameters):
        for i, (key, value) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(value)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][key]
            self.global_parameters[key] = tmp_v / len(local_parameters)


class Client:
    def __init__(self, data_loader, user_idx):
        self.data_loader = data_loader
        self.user_idx = user_idx

    def train(self, model, idx):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for i, (imgs, labels) in enumerate(self.data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            # print(f"Client: {idx}({self.user_idx:2d}) Local Epoch: [{local_epoch}][{i}/{len(self.data_loader)}]----loss {loss.item():.4f}")


def activate_client(train_dataloaders, user_idx, server):
    local_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]], user_idx[i]))
    return clients, local_parameters


def train(train_dataloaders, user_idx, server, global_model, up_link):
    clients, local_parameters = activate_client(train_dataloaders, user_idx, server)
    for local_epoch in range(1, local_epochs + 1):
        for i in range(len(user_idx)):
            model = net(in_channels, num_classes).to(device)
            model.load_state_dict(local_parameters[i])
            model.train()
            clients[i].train(model, i)
            local_parameters[i] = up_link.pass_link(model.to('cpu').state_dict())
        local_parameters_copy = copy.deepcopy(local_parameters)
        for cluster in range(num_clusters):
            cluster_idx = np.arange(cluster * num_per_cluster, (cluster + 1) * num_per_cluster)
            for j in range(num_per_cluster):
                for k, (key, value) in enumerate(local_parameters[cluster_idx[j]].items()):
                    tmp = torch.clone(value)
                    tmp += local_parameters_copy[cluster_idx[(j - 1) % num_per_cluster]][key]
                    tmp += local_parameters_copy[cluster_idx[(j + 1) % num_per_cluster]][key]
                    local_parameters[cluster_idx[j]][key] = tmp / 3

    server.upload(local_parameters)
    global_model.load_state_dict(server.global_parameters)


def valid(data_loader, model, epoch):
    with torch.no_grad():
        model.eval()
        losses = Recoder()
        accuracy = Recoder()
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            pred_labels = torch.softmax(logits, dim=1).argmax(dim=1)
            num_gold = (pred_labels == labels).sum() / len(imgs)
            losses.update(loss.item(), len(imgs))
            accuracy.update(num_gold.item(), len(imgs))
        print(f"Global Epoch: {epoch}----loss:{losses.avg():.4f}----accuracy:{accuracy.avg():.4f}")
    return accuracy.avg()


def train_main():
    seed_everything(42)
    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')
    if not os.path.exists(f'results/'):
        os.makedirs(f'results/')

    trainloaders, validloader = load_data(num_clients, data_dir)
    global_model = net(in_channels, num_classes)
    global_parameters = global_model.state_dict()
    up_link = Link("uplink")
    down_link = Link("downlink")
    server = FedAvgServer(global_parameters, down_link)

    accuracy_list = []
    for epoch in range(1, epochs + 1):
        user_idx = np.arange(num_clients)
        train(trainloaders, user_idx, server, global_model, up_link)
        test_model = copy.deepcopy(global_model).to(device)
        accuracy = valid(validloader, test_model, epoch)
        accuracy_list.append(accuracy)
        check_point(epoch, epochs, global_model, accuracy_list, save_model_interval, save_accuracy_interval)


if __name__ == '__main__':
    train_main()
