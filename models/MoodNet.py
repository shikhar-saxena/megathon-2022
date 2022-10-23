import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch.optim as optim


class MoodNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5)
        self.pool1 = nn.MaxPool2d(5, 2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.AvgPool2d(3, 2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.pool3 = nn.AvgPool2d(3, 2)
        
        self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc1 = nn.Linear(128, 1024)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(1024, out_channels)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool2(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.flatten(x)
        # x = torch.flatten(x)

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x

def test():
    x = torch.randn((10, 1, 48, 48)) # 10 samples of 48 x 48
    model = MoodNet()
    
    y = model(x)

    print(f'y = {y.shape}')
    print(f'x = {x.shape}')
    print()

# if __name__ == "__main__":
#     test()
