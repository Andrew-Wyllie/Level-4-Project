## 07/10/2023
## RESNET model for classical vs metal

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.models import ResNet
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd


#I dont have a gpu but include this line if others have one
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=100

##retriving data
os.listdir(r'C:\Users\andre\Documents\4th Year\Project\Level-4-Project\resnet_data\Data\genres_original')
base_dir = r'C:\Users\andre\Documents\4th Year\Project\Level-4-Project\resnet_data\Data\genres_original'
train_dir = r'C:\Users\andre\Documents\4th Year\Project\Level-4-Project\resnet_data\Data\images_original\train'
test_dir = r'C:\Users\andre\Documents\4th Year\Project\Level-4-Project\resnet_data\Data\images_original\test'

train_list = glob.glob(os.path.join(train_dir, '*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))

from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK']='True' ##To fix Error15 
train_list_len= len(train_list)
random_idx = np.random.randint(1,train_list_len, size=10)

#fig = plt.figure()

#i=1

#for idx in random_idx:
    #ax = fig.add_subplot(2,5,i)
   # img = Image.open(train_list[idx])
  #  plt.imshow(img)
 #   i+=1

#plt.axis('off')
#plt.show()

what_is_this = train_list[0].split('\\')[-1][:-9]

from sklearn.model_selection import train_test_split

train_list, val_list = train_test_split(train_list, test_size=0.2)

###################################################################

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class dataset(torch.utils.data.Dataset):

    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__ (self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('\\')[-1][:-9]
        if label == 'classical':
            label=1
        elif label == 'metal':
            label=0

        return img_transformed,label
    
train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

###########################################################################

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identifying_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 32
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identify_downsample = identifying_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2
        x = self.relu(x)        
        x = self.conv3(x)
        x = self.bn3
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module): #[3, 4, 6, 3] for ResNet50
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #RESNET LAYERS
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1) 
        self.layer2 = self._make_layer(block, layers[1], out_channels=124, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != (out_channels * 32):
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*32, kernel_size=1, stride=stride),nn.BatchNorm2d(out_channels*4))

        #this layer changes number of channels
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride)) 
        self.in_channels = out_channels*32

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) #256-> 64, 64*4 (256) again

        return nn.Sequential(*layers)
    

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [100, 4, 224, 224], img_channels, num_classes)

model = ResNet50().to(device)
model.train()

optimizer = torch.optim.Adam(params = model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()

        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss=0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1) == label.float().mean()))
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)

        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

classical_probs = []
model.eval()
with torch.no_grad():
    for data, fileid in test_loader:
        data = data.to(device)
        preds = model(data)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        classical_probs += list(zip(list(fileid), preds_list))

classical_probs.sort(key = lambda x : int(x[0]))
print(classical_probs)
idx = list(map(lambda x: x[0],classical_probs))
prob = list(map(lambda x: x[1],classical_probs))

submission = pd.DataFrame({'id':idx,'label':prob})
print(submission)

id_list = []
class_ = {0: 'metal', 1: 'classical'}

fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():
    
    i = random.choice(submission['id'].values)
    
    label = submission.loc[submission['id'] == i, 'label'].values[0]

    if label > 0.5:
        label = 1
    else:
        label = 0
        
    img_path = os.path.join(test_dir, '{}.png'.format(i))
    img = Image.open(img_path)
    
    ax.set_title(class_[label])
    ax.imshow(img)

print("rotors are good sir!")





#using a pretrained model
#resnet = torchvision.models.resnet34(pretrained=True)