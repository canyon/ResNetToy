import torch
import numpy as np
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from utils.cutout import Cutout

def draw_all_features(savenames):
    idx = 1
    fig_all = plt.figure()
    for savename in savenames:
        fig_all.add_subplot(2, 3, idx)
        plt.axis('off')
        plt.imshow(Image.open('utils/vis_pics/feature_map_{}.jpg'.format(savename)))
        idx += 1
    plt.tight_layout()
    fig_all.savefig('utils/vis_pics/feature_map.jpg', dpi=300)

def draw_features(x,savename):
    x = x.detach().cpu().numpy()
    width = 8
    height = 8
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
    plt.suptitle('Feature Map of {}'.format(savename), fontsize=25)
    fig.savefig('utils/vis_pics/feature_map_{}.jpg'.format(savename), dpi=300)

def draw_after_layer_img(tensor, name):
    tensor = tensor.squeeze(0)
    gray_scale = torch.sum(tensor,0)
    gray_scale = gray_scale / tensor.shape[0]
    gray_scale = gray_scale.data.cpu().numpy()
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(gray_scale)
    plt.axis("off")
    plt.title('Output Image of {}'.format(name), fontsize=12)
    fig.savefig('utils/vis_pics/output_{}.jpg'.format(name), dpi=300)

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save('utils/vis_pics/{}.jpg'.format(name))
    
def vis_feature_map(img, transforms, model):
    model.eval()
    img = transforms(Image.fromarray(np.uint8(np.load(img))))
    img = img.unsqueeze(0)

    f1 = model.conv1(img)
    print(f1.shape)
    # save_img(f1,'conv1')
    draw_features(f1, 'conv1')
    draw_after_layer_img(f1, 'conv1')

    new_model = nn.Sequential(*list(model.children())[:5])
    f3 = new_model(img)
    print(f3.shape)
    # save_img(f3, 'layer1')
    draw_features(f3, 'layer1')
    draw_after_layer_img(f3, 'layer1')
    
    new_model = nn.Sequential(*list(model.children())[:6])
    f4 = new_model(img)
    print(f4.shape)
    # save_img(f4, 'layer2')
    draw_features(f4, 'layer2')
    draw_after_layer_img(f4, 'layer2')
    
    new_model = nn.Sequential(*list(model.children())[:7])
    f5 = new_model(img)
    print(f5.shape)
    # save_img(f5, 'layer3')
    draw_features(f5, 'layer3')
    draw_after_layer_img(f5, 'layer3')
    
    new_model = nn.Sequential(*list(model.children())[:8])
    f6 = new_model(img)
    print(f6.shape)
    # save_img(f6, 'layer4')
    draw_features(f6, 'layer4')
    draw_after_layer_img(f6, 'layer4')
    
    draw_all_features(['conv1', 'layer1', 'layer2', 'layer3', 'layer4'])
    # plt.show()

if __name__ == '__main__':
    from utils.readData import read_dataset
    from utils.ResNet import ResNet18, ResNet34, ResNet50, ResNet101
    
    n_class = 10
    # batch_size = 128
    # train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size)
    img_path = 'C:/Users/79981/Desktop/CAS771/code/ResNetToy/dataset/10/test/0.npy'
    transform_train = transforms.Compose([
        # Fill with 0 around first, then randomly crop the image into 32*32
        transforms.RandomCrop(32, padding=4), 
        # The image has half the probability of flipping and half of the probability of not flipping
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # The mean and variance used in the normalization of each layer of R, G, and B
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
        Cutout(n_holes=1, length=16),
    ])

    # Get the pre-trained model
    model = ResNet34()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify the last fully connected layer
    model.fc = torch.nn.Linear(512, n_class)
    # load weight
    model.load_state_dict(torch.load('checkpoint/run7/resnet34_class=10_step=300_batch=32_kernel=3_dataAug=yes.pt'))

    vis_feature_map(img_path, transform_train, model)