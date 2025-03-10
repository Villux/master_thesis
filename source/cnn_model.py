import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data

from data_loader import load_data
from utils.helpers import create_h5py_dataset_with_cache
from models.fcn import FCN
# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 100
LR = 0.001
USE_CUDA = torch.cuda.is_available()

n_classes = 3
n_channels = 2

VARIANCE_ID = 1
train_dataset = create_h5py_dataset_with_cache("training", BATCH_SIZE)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

val_dataset = create_h5py_dataset_with_cache("validation", BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = create_h5py_dataset_with_cache("test", BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


fcn = FCN(n_channels)

print(f"MODEL TRAINABLE PARAMETERS: {sum(p.numel() for p in fcn.parameters() if p.requires_grad)}")

if USE_CUDA:
    print("CUDA USED")
    fcn = fcn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)

# Train the Model
for epoch in range(EPOCH):
    for i, data in enumerate(train_loader):
        tseries = Variable(data['data']).float()
        labels = Variable(data['label']).type(torch.LongTensor)
        if USE_CUDA:
            tseries = tseries.cuda()
            labels = labels.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = fcn(tseries)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, EPOCH, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))


fcn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for i, data in enumerate(train_loader):
    tseries = Variable(data['data']).float()
    labels = Variable(data['label']).type(torch.LongTensor)
    if USE_CUDA:
            tseries = tseries.cuda()
            labels = labels.cuda()
    outputs = fcn(tseries)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Train Accuracy of the model: %d %%' % (100 * correct / total))

correct = 0
total = 0
for i, data in enumerate(test_loader):
    tseries = Variable(data['data']).float()
    labels = Variable(data['label']).type(torch.LongTensor)
    if USE_CUDA:
            tseries = tseries.cuda()
            labels = labels.cuda()
    outputs = fcn(tseries)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Test Accuracy of the model: %d %%' % (100 * correct / total))
