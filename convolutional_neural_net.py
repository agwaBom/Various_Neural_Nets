# import
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# model
class ConvolutionalNN(nn.Module):
    def __init__(self, input_size, input_channel, output_size):
        super(ConvolutionalNN, self).__init__() 
        hidden_channel_1 = 10
        hidden_channel_2 = 5
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=hidden_channel_1, kernel_size=3, stride=1)
        output = self.calculate_output_size(input_size, 3, 0, 1)

        self.maxpooling1 = nn.MaxPool2d(3, stride=1)
        output = self.calculate_output_size(output, 3, 0, 1)

        self.conv2 = nn.Conv2d(in_channels=hidden_channel_1, out_channels=hidden_channel_2, kernel_size=3, stride=1)
        output = self.calculate_output_size(output, 3, 0, 1)

        self.maxpooling2 = nn.MaxPool2d(3, stride=1)
        output = self.calculate_output_size(output, 3, 0, 1)
        output = int(output)

        self.linear = nn.Linear(in_features=hidden_channel_2*output*output, out_features=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.maxpooling1(x))
        x = self.conv2(x)
        x = self.relu(self.maxpooling2(x))
        # flattening
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def calculate_output_size(self, input_size, kernel_size, padding, stride):
        # [(input_size−Kernel+2Padding)/Stride]+1.
        output_size = ((input_size-kernel_size+2*padding)/stride)+1
        return output_size

# load data & hyperparameter
input_size = 28
input_channel = 1
output_size = 10
lr = 0.0001
n_epochs = 50

model = ConvolutionalNN(input_size, input_channel, output_size)

# model testing
# x = torch.randn(10, 1, 28, 28)
# print(model(x).size())

train_data = datasets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_iter = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

valid_data = datasets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
valid_iter = DataLoader(dataset=valid_data, batch_size=32, shuffle=True)

# loss & gradient descent
loss_function = nn.CrossEntropyLoss()
creterion = optim.Adam(model.parameters(), lr=lr)

# train loop
for epoch in range(0, n_epochs):
    print(f"### EPOCH {epoch} ###")
    print("Training...")
    for data, target in tqdm(train_iter):
        model.train()
        creterion.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        creterion.step()
    print(f"Train Loss : [{loss}]")
    
    correct = 0
    total = 0
    print("Validating...")
    for data, target in tqdm(valid_iter):
        model.eval()
        output = model(data)
        loss = loss_function(output, target)
        prediction = output.max(1)
        correct += prediction.indices.eq(target).sum()
        total += target.size(0)

    print(f"Validation Accuracy : [{correct/total}], Loss : [{loss}]\n")

