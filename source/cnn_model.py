import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data

from data_loader import load_data

# Load data
Xtrain, ytrain, Xval, yval, Xtest, ytest, n_classes = load_data()

# Hyper Parameters
EPOCH = 40
BATCH_SIZE = 100
LR = 0.001

n_classes = n_classes
n_channels = 2

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain).double(), torch.from_numpy(ytrain).double())
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 5, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2))
        self.fc = nn.Linear(60*5, n_classes)

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
    for i, (x, y) in enumerate(train_loader):
        tseries = Variable(x).float()
        labels = Variable(y).type(torch.LongTensor)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(tseries)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, EPOCH, i+1, len(Xtrain)//BATCH_SIZE, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for x, y in val_loader:
    tseries = Variable(x).float()
    labels = Variable(y).type(torch.LongTensor)
    outputs = cnn(tseries)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Validation Accuracy of the model: %d %%' % (100 * correct / total))
