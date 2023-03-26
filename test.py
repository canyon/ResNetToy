import torch
import torch.nn as nn
from tqdm import tqdm
from utils.readData import read_dataset
from utils.ResNet import ResNet18, ResNet34, ResNet50, ResNet101
# set device
device = 'cuda'
n_class = 10
batch_size = 16
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size)
# Get the pre-trained model
model = ResNet34()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
# Modify the last fully connected layer
model.fc = torch.nn.Linear(512, n_class)
# load weight
model.load_state_dict(torch.load('checkpoint/run7/resnet34_class=10_step=300_batch=32_kernel=3_dataAug=yes.pt'))
model = model.to(device)

total_sample = 0
right_sample = 0
# validate model
model.eval()
for data, target in tqdm(test_loader):
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    # correct = np.squeeze(correct_tensor.to(device).numpy())
    total_sample += batch_size
    for i in correct_tensor:
        if i:
            right_sample += 1
print('Accuracy: {:.6f}%'.format(100*right_sample/total_sample))