import torch
import torch.nn as nn

conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)
maxpooling1 = nn.MaxPool2d(3, stride=1)
relu = nn.ReLU()

input = torch.randn(20, 1, 28, 28)


x = conv1(input)
print(x.shape)
x = maxpooling1(x)
print(x.shape)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input.size())
print(target.size())
output = loss(input, target)