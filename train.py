import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18, ResNet34, ResNet50, ResNet101
from utils.arguments import get_args


# Add seed for reproducible result
# Added starts from run5
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# # Disable automatic selection of best method
# torch.backends.cudnn.benchmark = False
# # Only use deterministic algorithms
# torch.backends.cudnn.deterministic = True

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# set device
device = 'cuda'
torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()


##################################== Edit Below ==###############################
kernel_size = 3
# read dataset
batch_size = 128
n_class = 100
# 50 epochs/h on 3060
n_epochs = 50*4

## 第几次run就改成几
runs = 'run1'
resnet_level = '18'
# Load the model (use preprocessed model, modify last layer, fix previous weights)
model = ResNet18()
#################################################################################

saved_model_name = 'resnet{}_class={}_step={}_batch={}_kernel={}_dataAug=yes'.format(resnet_level, n_class, n_epochs, batch_size, kernel_size)
# Import TensorBoard library and set up a SummaryWriter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/{}-{}'.format(runs, saved_model_name))



save_model_path = os.path.join(os.getcwd(), 'checkpoint', runs)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size)
"""
The 7x7 downsampling convolution and pooling operations of the ResNet18 network tend to lose part of the information.
So in the experiment we removed the 7x7 downsampling layer and the maximum pooling layer and replaced it with a 3x3 downsampling convolution.
Reduce the stride and padding size of this convolutional layer at the same time
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, stride=1, padding=1, bias=False)

# Change the last fully connected layer
no_bottleneck = 512
yes_bottleneck = 512 * 4

##################################== Edit Below ==###############################
## resnet50 change to yes_bottleneck
model.fc = torch.nn.Linear(no_bottleneck, n_class)
#################################################################################

model = model.to(device)
# Use the cross entropy loss function
criterion = nn.CrossEntropyLoss().to(device)

# start training
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.1
counter = 0
tqdm_list = tqdm(range(1, n_epochs+1))
for epoch in tqdm_list:
    # resnet50 needs ~10g memery
    torch.cuda.empty_cache()

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
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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
    tqdm_list.set_description('Epoch: [{}/{}]'.format(epoch, n_epochs))
    tqdm_list.set_postfix({'Accuracy': '{:.6f}'.format(100*tmp_acc), 'Training Loss': '{:.6f}'.format(train_loss), 'Validation Loss': '{:.6f}'.format(valid_loss)})
    
    # If the validation set loss function decreases, save the model
    if valid_loss <= valid_loss_min:
        print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # torch.save(model.state_dict(), 'checkpoint/{}/{}.pt'.format(runs, saved_model_name))
        torch.save(model.state_dict(), os.path.join(save_model_path, saved_model_name))
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', valid_loss, epoch)
    writer.add_scalar('Accuracy', 100*tmp_acc, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
writer.close()


# if __name__ == '__main__':
#   args = get_args()
#   print('Called with args:')
#   print(args)

