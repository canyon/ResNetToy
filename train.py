import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Import TensorBoard library and set up a SummaryWriter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='log_dir/run1')

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# read dataset
batch_size = 128
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size)
# Load the model (use preprocessed model, modify last layer, fix previous weights)
n_class = 10
model = ResNet18()
"""
The 7x7 downsampling convolution and pooling operations of the ResNet18 network tend to lose part of the information.
So in the experiment we removed the 7x7 downsampling layer and the maximum pooling layer and replaced it with a 3x3 downsampling convolution.
Reduce the stride and padding size of this convolutional layer at the same time
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
# Change the last fully connected layer
model.fc = torch.nn.Linear(512, n_class)
model = model.to(device)
# Use the cross entropy loss function
criterion = nn.CrossEntropyLoss().to(device)

# start training
n_epochs = 10
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    # Dynamically adjust the learning rate
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #########################
    # model on training set #
    #########################
    # To enable batch normalization and drop out
    model.train()
    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        #（equals to: output = model.forward(data).to(device) ）
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ###########################
    # Model on validation set #
    ###########################

    # validate model
    model.eval()
    for data, target in tqdm(valid_loader):
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    tmp_acc = right_sample/total_sample
    accuracy.append(tmp_acc)
 
    # Calculate average loss
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # Show loss functions for training and validation sets
    epoch.set_description('Epoch: [{}/{}]'.format(epoch, n_epochs))
    epoch.set_postfix('Accuracy: {:.6f} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(100*tmp_acc, train_loss, valid_loss))
    
    # If the validation set loss function decreases, save the model
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', valid_loss, epoch)
    writer.add_scalar('Accuracy', 100*tmp_acc, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
writer.close()