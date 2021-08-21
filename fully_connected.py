# import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# model

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear_1 = nn.Linear(input_size, 512)
        self.linear_2 = nn.Linear(512, 128)
        self.linear_3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, input_feature):
        x = self.linear_1(input_feature)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)

        return x

# accuracy
def test_data(data_loader, model):
    model.eval()
    
    accuracy = 0
    total_data = 0
    for data, target in data_loader:
        data = data.contiguous().view(data.size(0), -1)
        output = model(data)        

        prediction = torch.max(output, dim=1)
        accuracy += prediction.indices.eq(target).sum()
        total_data += prediction.indices.size(0)

    return accuracy/total_data

# hyperparameter
batch_size = 256
num_epochs = 50
learning_rate = 0.0001
num_classes = 10
input_size = 784

# load dataset & model
train_dataset = datasets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = SimpleNN(input_size=input_size, output_size=num_classes)

# loss & optimizer
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

# train valid loop
for epoch in range(0, num_epochs):
    print(f"### EPOCH : {epoch} ###")
    # train loop
    for data, target in tqdm(train_dataloader):
        model.train()
        optimizer.zero_grad()
        data = data.contiguous().view(data.size(0), -1)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    accuracy = test_data(test_dataloader, model)
    print(f"Accuracy : [{accuracy}] loss: [{loss}]\n")
    