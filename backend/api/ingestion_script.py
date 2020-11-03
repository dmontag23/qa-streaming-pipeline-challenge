from collections import OrderedDict

import requests
import torch
import torch.nn as nn
import torch.optim as optim

from neural_network import Network

page = 0
increment = 1000
net = Network()

def validate_model(model, dataloader):
    total = 0.0
    num_correct = 0.0
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        num_correct = num_correct + ((torch.round(outputs) - labels) == 0).sum().item()
        total = total + labels.numel()

    accuracy = 100 * num_correct / total
    return accuracy

def train_neural_network(nn_input, nn_target):
    global net
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(nn_input), torch.FloatTensor(nn_target))
    train_size = int(0.8 * len(dataset))
    validate_size = len(dataset) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
    dataloaders = OrderedDict([
        ('train', torch.utils.data.DataLoader(train_dataset, batch_size=increment * 10, shuffle=True)),
        ('validate', torch.utils.data.DataLoader(validate_dataset, batch_size=increment * 10, shuffle=True))
    ])
    alpha = .1
    loss_funct = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=alpha)
    dataloader = dataloaders['train']
    for epoch in range(100):  # loop over the dataset multiple times
        print(epoch)
        for ii, (inputs, labels) in enumerate(dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # run the inputs through the nn to get the nn output and compute the loss function
            outputs = net(inputs)
            loss = loss_funct(outputs, labels)

            # update the parameters of the nn
            loss.backward()
            optimizer.step()


    accuracy = validate_model(net, dataloaders['validate'])
    print("Test Accuracy: %.2f" %accuracy)
    return accuracy

def run_ingestion_script():
    global page
    nn_input = []
    nn_target = []
    for i in range(page, page + increment):
        request = requests.get('http://localhost:5000/api/v1/data?page=' + str(i)).json()
        for data_point in request:  # iterate through the data received by the request
            competence = data_point['competence']
            network_ability = data_point['network_ability']
            nn_input.append([competence, network_ability])
            promoted = data_point['promoted']
            nn_target.append([promoted])

    page = page + increment
    accuracy = train_neural_network(nn_input, nn_target)
    return accuracy


