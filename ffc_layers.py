####Binary classifcation between classical and metal with 4 fully connected layers 

import torch

# fully connected layers in pytorch -> linear layer

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


#I dont have a gpu but i included this line if others have one
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

fig = plt.figure()

i=1

for idx in random_idx:
    ax = fig.add_subplot(2,5,i)
    img = Image.open(train_list[idx])
    plt.imshow(img)
    i+=1

plt.axis('off')
#plt.show()


#what_is_this = train_list[0].split('\\')[-1][:-9]

##############################################

###FULLY CONNECTED LAYER

##############################################

class NN(nn.Module):
    def __init__(self, input_size, num_classes): #dataset [432x288]
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            #nn.ReLU(),
            nn.Linear(512, 256),
            #nn.ReLU(),
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

input_size = 432*288*4
num_classes = 2
learning_rate = 0.01
batch_size = 10
epochs = 5

train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
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

train_data = dataset(train_list, transform = train_transforms)
test_data = dataset(test_list, transform = test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)
model.train()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        #[batch_size, 4, width, height]
        output = model(data)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()

        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))


classical_probs = []
model.eval()
with torch.no_grad():
    for data, fileid in test_loader:
        data = data.to(device)
        preds = model(data)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        classical_probs += list(zip(list(fileid), preds_list))

classical_probs.sort(key = lambda x : int(x[0]))
idx = list(map(lambda x: x[0],classical_probs))
prob = list(map(lambda x: x[1],classical_probs))

print("idx ", idx)
print("prob ", prob)

for x in range(len(prob)):
    print("x =>", prob[x])
    if prob[x] > 0.5:
        prob[x] = 'classical'
    else:
        prob[x] = 'metal'

print("prob", prob)

submission = pd.DataFrame({'id':idx,'label':prob})
print(submission)

print('rotors are good sir')