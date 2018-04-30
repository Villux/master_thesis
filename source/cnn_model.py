import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data

from data_loader import load_data
from utils.helpers import create_h5py_dataset_with_cache
# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 100
LR = 0.001

n_classes = 3
n_channels = 2

VARIANCE_ID = 1
train_dataset = create_h5py_dataset_with_cache("training", BATCH_SIZE)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

val_dataset = create_h5py_dataset_with_cache("validation", BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = create_h5py_dataset_with_cache("test", BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_channels, 18, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2))
        self.fc = nn.Linear(58 * 4, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn = CNN()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# Train the Model
for epoch in range(EPOCH):
    for i, data in enumerate(train_loader):
        tseries = Variable(data['data']).float()
        labels = Variable(data['label']).type(torch.LongTensor)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(tseries)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, EPOCH, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for i, data in enumerate(val_loader):
    tseries = Variable(data['data']).float()
    labels = Variable(data['label']).type(torch.LongTensor)
    outputs = cnn(tseries)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Validation Accuracy of the model: %d %%' % (100 * correct / total))

correct = 0
total = 0
for i, data in enumerate(test_loader):
    tseries = Variable(data['data']).float()
    labels = Variable(data['label']).type(torch.LongTensor)
    outputs = cnn(tseries)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Test Accuracy of the model: %d %%' % (100 * correct / total))
