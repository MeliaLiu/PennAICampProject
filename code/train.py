from dataset import HieroglyphicsDataset
from net import MyNet

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

def train(model, train_loader, criterion, optimizer, epochs):
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))

                    running_loss = 0.0

# Hyper parameters
lr = 0.01
momentum = 0.9
epochs = 50

load_path = None # path_to_load
save_path = './models/new_model.pth' # path to save

model = MyNet(category_num=9)
if load_path != None:
    model.load_state_dict(torch.load(load_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

trainset = HieroglyphicsDataset('./board_dataset', 
                                'train', 
                                transform=transforms.Compose
                                (
                                    [
                                        transforms.Resize((224, 224))
                                    ]
                                )
                               )

train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True)

train(model, train_dataloader, criterion, optimizer, epochs)

torch.save(model.state_dict(), save_path)