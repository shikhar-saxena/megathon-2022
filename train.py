import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm

# import cv2 # required by albumentations library for flags to be parsed

import torchvision

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import torch.optim as optim

from models.MoodNet import MoodNet


import os
from matplotlib import pyplot as plt
import numpy as np

IMG_HEIGHT = 48
IMG_WIDTH = 48

batch_size = 32

train_dir = 'data/train'
test_dir = 'data/test'

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
]))

test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
]))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = train_data.classes
print(f'classes: {classes}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = MoodNet()
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

epochs = 30

history = []
train_losses = []
val_losses = []

train_accuracy = []
val_accuracy = []

gen_error = []


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(batch):
    images, labels = batch 

    images = images.to(device)
    labels = labels.to(device)

    # print(labels)

    out = net(images)                   # Generate predictions
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy

    return [loss, acc]

def epoch_end_output(outputs):
    if not torch.is_tensor(outputs):
        outputs = torch.tensor(outputs)

    batch_losses = outputs[:, 0]
    epoch_loss = batch_losses.mean()   # Combine losses

    batch_accs = outputs[:, 1]
    epoch_acc = batch_accs.mean()      # Combine accuracies

    return [epoch_loss.item(), epoch_acc.item()]

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    
    outputs = [validation_step(batch) for batch in val_loader]
    return epoch_end_output(outputs)

def epoch_end_result(epoch, result):
        print("Epoch [{}], avg_train_loss: {:.4f}, val_loss: {:.4f}, avg_train_acc: {:.4f}, val_acc: {:.4f}, gen_err: {:.4f}".format(epoch, result[0], result[1], result[2], result[3], result[4]))

for epoch in range(epochs):
    loss_avg = 0
    outputs = []
    for batch in train_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        preds = net(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()

        loss_avg += loss.item()
        acc = accuracy(preds, labels)

        loss.backward()
        optimizer.step()

        outputs.append([loss.item(), acc.item()])

    train_loss, train_acc = epoch_end_output(outputs)

    # Validation phase
    with torch.no_grad():
        # Your eval code
        val_loss, val_acc = evaluate(net, test_loader)

    train_losses.append(train_loss)
    train_accuracy.append(train_acc)

    val_losses.append(val_loss)
    val_accuracy.append(val_acc)

    gen_err = train_acc - val_acc
    gen_error.append(gen_err)

    history.append([train_loss, val_loss, train_acc, val_acc, gen_err])
    epoch_end_result(epoch, history[-1])


print('Finished Training')

PATH = './moodnet.pth'
torch.save(net.state_dict(), PATH)

# Test the network on the test data
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

