 ####5 genre classification

import torch

# fully connected layers in pytorch -> linear layer

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import torchvision
from torchvision import datasets, transforms
from torchvision.models import ResNet
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
from statistics import mean



start_time = datetime.utcnow()
print("Starting at: ", start_time)

#I dont have a gpu but i included this line if others have one
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################
from google.colab import drive
drive.mount('/content/gdrive')
######################################################

classes = ["blues", "classical", "hiphop", "metal", "hiphop"]

#Hyperparameters
input_size = 218*336*4
num_classes = 5 ##genres to be seperated
learning_rate = 0.01
batch_size = 10
epochs = 50

##retriving data
train_dir = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/train5'
test_dir = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/test5'

train_list = glob.glob(os.path.join(train_dir, '*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))

#BLUES VS CLASSICAL
train_dirbcl = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainbcl'
test_dirbcl = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testbcl'

train_listbcl = glob.glob(os.path.join(train_dirbcl, '*.png'))
test_listbcl = glob.glob(os.path.join(test_dirbcl, '*.png'))

#BLUES VS HIPHOP
train_dirbco = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainbh'
test_dirbco = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testbh'

train_listbco = glob.glob(os.path.join(train_dirbco, '*.png'))
test_listbco = glob.glob(os.path.join(test_dirbco, '*.png'))

#BLUES VS metal
train_dirbd = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainbm'
test_dirbd = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testbm'

train_listbd = glob.glob(os.path.join(train_dirbd, '*.png'))
test_listbd = glob.glob(os.path.join(test_dirbd, '*.png'))

#BLUES VS POP
train_dirbh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainbp'
test_dirbh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testbp'

train_listbh = glob.glob(os.path.join(train_dirbh, '*.png'))
test_listbh = glob.glob(os.path.join(test_dirbh, '*.png'))

#CLASSICAL VS HIPHOP
train_dirclco = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainclh'
test_dirclco = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testclh'

train_listclco= glob.glob(os.path.join(train_dirclco, '*.png'))
test_listclco = glob.glob(os.path.join(test_dirclco, '*.png'))

#CLASSICAL VS METAL
train_dircld = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainclm'
test_dircld = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testclm'

train_listcld = glob.glob(os.path.join(train_dircld, '*.png'))
test_listcld = glob.glob(os.path.join(test_dircld, '*.png'))

#CLASSICAL VS POP
train_dirclh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainclp'
test_dirclh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testclp'

train_listclh = glob.glob(os.path.join(train_dirclh, '*.png'))
test_listclh = glob.glob(os.path.join(test_dirclh, '*.png'))

#hiphop VS metal
train_dircod = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainhm'
test_dircod = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testchm'

train_listcod = glob.glob(os.path.join(train_dircod, '*.png'))
test_listcod = glob.glob(os.path.join(test_dircod, '*.png'))

#hiphop VS POP
train_dircoh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainhp'
test_dircoh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testchp'

train_listcoh = glob.glob(os.path.join(train_dircoh, '*.png'))
test_listcoh = glob.glob(os.path.join(test_dircoh, '*.png'))

#metal VS pop
train_dirdh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/trainmp'
test_dirdh = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/testmp'

train_listdh = glob.glob(os.path.join(train_dirdh, '*.png'))
test_listdh = glob.glob(os.path.join(test_dirdh, '*.png'))

# I want to look at seperating the training
# data into validation data in the
# future maybe as an inprovement

from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK']='True' ##To fix Error15
#random_idx = np.random.randint(1,len(train_list), size=10)

#what_is_this = train_list[0].split('\\')[-1][:-9]

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

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'classical':
            label = 1
        elif label == 'hiphop':
            label = 2
        elif label == 'metal':
            label = 3
        elif label == 'pop':
            label = 4

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)


        return img_transformed,label

class datasetbcl(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'classical':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetbco(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'hiphop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetbd(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'metal':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetbh(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'pop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label#

class datasetclco(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'classical':
            label = 0
        elif label == 'hiphop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetcld(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'classical':
            label = 0
        elif label == 'metal':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetclh(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'classical':
            label = 0
        elif label == 'pop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetcod(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'hiphop':
            label = 0
        elif label == 'metal':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetcoh(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'hiphop':
            label = 0
        elif label == 'pop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

class datasetdh(torch.utils.data.Dataset):

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'metal':
            label = 0
        elif label == 'pop':
            label = 1

        img_transformed = transforms.functional.crop(img=img_transformed, top=35, left=54, height=218, width=336)

        return img_transformed,label

train_data = dataset(train_list, transform = train_transforms)
test_data = dataset(test_list, transform = test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

###blues v classical
train_databcl = datasetbcl(train_listbcl, transform = train_transforms)
test_databcl = datasetbcl(test_listbcl, transform = test_transforms)

train_loaderbcl = torch.utils.data.DataLoader(dataset = train_databcl, batch_size=batch_size, shuffle=True)
test_loaderbcl = torch.utils.data.DataLoader(dataset = test_databcl, batch_size=batch_size, shuffle=True)

###blues v hiphop
train_databco = datasetbco(train_listbco, transform = train_transforms)
test_databco = datasetbco(test_listbco, transform = test_transforms)

train_loaderbco = torch.utils.data.DataLoader(dataset = train_databco, batch_size=batch_size, shuffle=True)
test_loaderbco = torch.utils.data.DataLoader(dataset = test_databco, batch_size=batch_size, shuffle=True)

###blues v metal
train_databd = datasetbd(train_listbd, transform = train_transforms)
test_databd = datasetbd(test_listbd, transform = test_transforms)

train_loaderbd = torch.utils.data.DataLoader(dataset = train_databd, batch_size=batch_size, shuffle=True)
test_loaderbd = torch.utils.data.DataLoader(dataset = test_databd, batch_size=batch_size, shuffle=True)

###blues v hiphop
train_databh = datasetbh(train_listbh, transform = train_transforms)
test_databh = datasetbh(test_listbh, transform = test_transforms)

train_loaderbh = torch.utils.data.DataLoader(dataset = train_databh, batch_size=batch_size, shuffle=True)
test_loaderbh = torch.utils.data.DataLoader(dataset = test_databh, batch_size=batch_size, shuffle=True)

###classical v hiphop
train_dataclco = datasetclco(train_listclco, transform = train_transforms)
train_loaderclco = torch.utils.data.DataLoader(dataset = train_dataclco, batch_size=batch_size, shuffle=True)

###classical v metal
train_datacld = datasetcld(train_listcld, transform = train_transforms)
train_loadercld = torch.utils.data.DataLoader(dataset = train_datacld, batch_size=batch_size, shuffle=True)

###classical v hiphop
train_dataclh = datasetclh(train_listclh, transform = train_transforms)
train_loaderclh = torch.utils.data.DataLoader(dataset = train_dataclh, batch_size=batch_size, shuffle=True)

###hiphop v metal
train_datacod = datasetcod(train_listcod, transform = train_transforms)
train_loadercod = torch.utils.data.DataLoader(dataset = train_datacod, batch_size=batch_size, shuffle=True)

###hiphop v hiphop
train_datacoh = datasetcoh(train_listcoh, transform = train_transforms)
train_loadercoh = torch.utils.data.DataLoader(dataset = train_datacoh, batch_size=batch_size, shuffle=True)

###metal v hiphop
train_datadh = datasetdh(train_listdh, transform = train_transforms)
train_loaderdh = torch.utils.data.DataLoader(dataset = train_datadh, batch_size=batch_size, shuffle=True)

for (image, label) in train_loader:
    print("ITS WORKING*************************************************************8")
    print(image[0])
    plt.imshow(image[0].squeeze().permute(1,2,0))
    plt.show()

def calculate_accuracy(model, dataloader):
    num_correct = 0
    num_samples = 0
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).argmax(dim=1)
            num_correct += (y_pred == y).sum().item()
            num_samples += y.size(0)

    model.train() # Set model back to training mode
    return num_correct / num_samples

##############################################
###MODELS
##############################################


class FNN(nn.Module):
    def __init__(self, input_size, num_classes): #dataset [432x288]
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Linear(100, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class FNN2(nn.Module):
    def __init__(self, input_size, num_classes): #dataset [432x288]
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

##############################################
###TRAINING
##############################################

##MAIN

model = FNN(input_size=input_size, num_classes=num_classes).to(device)
model.train(True)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

final_loss = 0
final_accuracy = 0

loss_list = []
accuracy_list = []

epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1


    epoch_accuracy = calculate_accuracy(model, train_loader)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)
    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss

print('Finished Training')

##BLUES VS CLASSICAL
modelbcl = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelbcl.train(True)
print(modelbcl)

criterion = nn.CrossEntropyLoss()
optimizerbcl = torch.optim.Adam(modelbcl.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderbcl:
        data = data.to(device)
        label = label.to(device)

        output = modelbcl(data)
        loss = criterion(output, label)

        optimizerbcl.zero_grad()
        loss.backward()
        optimizerbcl.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelbcl, train_loaderbcl)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelbcl.train(False)

##BLUES VS hiphop
modelbco = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelbco.train(True)
print(modelbco)

criterionbco = nn.CrossEntropyLoss()
optimizerbco = torch.optim.Adam(modelbco.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderbco:
        data = data.to(device)
        label = label.to(device)

        output = modelbco(data)
        loss = criterionbco(output, label)

        optimizerbco.zero_grad()
        loss.backward()
        optimizerbco.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelbco, train_loaderbco)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
       break
    last_loss = loss


print('Finished Training')
modelbco.train(False)

##BLUES VS metal
modelbd = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelbd.train(True)
print(modelbd)

criterionbd = nn.CrossEntropyLoss()
optimizerbd = torch.optim.Adam(modelbd.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderbd:
        data = data.to(device)
        label = label.to(device)

        output = modelbd(data)
        loss = criterionbd(output, label)

        optimizerbd.zero_grad()
        loss.backward()
        optimizerbd.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelbd, train_loaderbd)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelbd.train(False)

##BLUES VS pop
modelbh = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelbh.train(True)
print(modelbh)

criterionbh = nn.CrossEntropyLoss()
optimizerbh = torch.optim.Adam(modelbh.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderbh:
        data = data.to(device)
        label = label.to(device)

        output = modelbh(data)
        loss = criterionbh(output, label)

        optimizerbh.zero_grad()
        loss.backward()
        optimizerbh.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelbh, train_loaderbh)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelbh.train(False)

##CLASSICAL VS hiphop
modelclco = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelclco.train(True)
print(modelclco)

criterionclco = nn.CrossEntropyLoss()
optimizerclco = torch.optim.Adam(modelclco.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderclco:
        data = data.to(device)
        label = label.to(device)

        output = modelclco(data)
        loss = criterionclco(output, label)

        optimizerclco.zero_grad()
        loss.backward()
        optimizerclco.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelclco, train_loaderclco)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelclco.train(False)

##classical VS metal
modelcld = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelcld.train(True)
print(modelcld)

criterioncld = nn.CrossEntropyLoss()
optimizercld = torch.optim.Adam(modelcld.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loadercld:
        data = data.to(device)
        label = label.to(device)

        output = modelcld(data)
        loss = criterioncld(output, label)

        optimizercld.zero_grad()
        loss.backward()
        optimizercld.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelcld, train_loadercld)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelcld.train(False)

##classical VS pop
modelclh = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelclh.train(True)
print(modelclh)

criterionclh = nn.CrossEntropyLoss()
optimizerclh = torch.optim.Adam(modelclh.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderclh:
        data = data.to(device)
        label = label.to(device)

        output = modelclh(data)
        loss = criterionclh(output, label)

        optimizerclh.zero_grad()
        loss.backward()
        optimizerclh.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelclh, train_loaderclh)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelclh.train(False)

##hiphop VS metal
modelcod = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelcod.train(True)
print(modelcod)

criterioncod = nn.CrossEntropyLoss()
optimizercod = torch.optim.Adam(modelcod.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loadercod:
        data = data.to(device)
        label = label.to(device)

        output = modelcod(data)
        loss = criterioncod(output, label)

        optimizercod.zero_grad()
        loss.backward()
        optimizercod.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelcod, train_loadercod)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss


print('Finished Training')
modelcod.train(False)

##hiphop VS pop
modelcoh = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modelcoh.train(True)
print(modelcoh)

criterioncoh = nn.CrossEntropyLoss()
optimizercoh = torch.optim.Adam(modelcoh.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loadercoh:
        data = data.to(device)
        label = label.to(device)

        output = modelcoh(data)
        loss = criterioncoh(output, label)

        optimizercoh.zero_grad()
        loss.backward()
        optimizercoh.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modelcoh, train_loadercoh)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    ###Dog leg - reducing dimishing returns
    #if loss > 2*last_loss:
      #break

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss

print('Finished Training')
modelcoh.train(False)

##metal VS pop
modeldh = FNN2(input_size=input_size, num_classes=num_classes).to(device)
modeldh.train(True)
print(modeldh)

criteriondh = nn.CrossEntropyLoss()
optimizerdh = torch.optim.Adam(modeldh.parameters(), lr=learning_rate)


epoch_accuracy = 0
epoch_loss = 0
last_loss = 0

for epoch in range(epochs):
    count = 0

    for data, label in train_loaderdh:
        data = data.to(device)
        label = label.to(device)

        output = modeldh(data)
        loss = criteriondh(output, label)

        optimizerdh.zero_grad()
        loss.backward()
        optimizerdh.step()
        count += 1


    epoch_accuracy = calculate_accuracy(modeldh, train_loaderdh)
    epoch_loss = loss

    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_accuracy)

    print("Epoch , {}/{}, Training Accuracy: {}, Training Loss: {}".format(epoch+1, epochs, epoch_accuracy , epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)

    if (epoch_accuracy > 0.9) & (loss < 1):
      break

    ###Stops overfitting
    if ((loss_diff) < 0.05) & (epoch > 5):
       break
    if loss == 0:
      break
    last_loss = loss

print('Finished Training')
modeldh.train(False)

##############################################
###TESTING
##############################################

predicted_list = []
from functools import reduce
true_list = []
model.eval()
modelbcl.eval()
modelbco.eval()
modelbd.eval()
modelbh.eval()
modelclco.eval()
modelcld.eval()
modelclh.eval()
modelcod.eval()
modelcoh.eval()
modeldh.eval()
total_bi_pred = []
bi_pred = []
bipredicted_list = []
final_list = []
n_correct = 0
bi_correct = 0
fin_correct = 0
final_correct = 0
final_pred = []

def get_conf(outputs):
  prob = torch.softmax(outputs, dim=1)
  top_p, top_class = prob.topk(1, dim = 1)
  confidence = top_p[0]
  print("confidence =", confidence)
  return confidence.item()
from sklearn.metrics import f1_score

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputbcl = modelclco(images)
        outputbco = modelbco(images)
        outputbd = modelbd(images)
        outputbh = modelbh(images)
        outputclco = modelclco(images)
        outputcld = modelcld(images)
        outputclh = modelclh(images)
        outputcod = modelcod(images)
        outputcoh = modelcoh(images)
        outputdh = modeldh(images)

      ###I could run softmax for all outputs and then using torch.max insteadof _ print confidence

        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        confidence = get_conf(outputs)

        _, predictedbcl = torch.max(outputbcl.data, 1)
        predictedbcl = predictedbcl.tolist()
        for i in range(len(predictedbcl)):
          if predictedbcl[i] == 0:
            predictedbcl[i] = 0
          elif predictedbcl[i] == 1:
            predictedbcl[i] = 1
        confidencebcl = get_conf(outputbcl)

        _, predictedbco = torch.max(outputbco.data, 1)
        predictedbco = predictedbco.tolist()
        for i in range(len(predictedbco)):
          if predictedbco[i] == 0:
            predictedbco[i] = 0
          elif predictedbco[i] == 1:
            predictedbco[i] = 2
        confidencebco = get_conf(outputbco)

        _, predictedbd = torch.max(outputbd.data, 1)
        predictedbd = predictedbd.tolist()
        for i in range(len(predictedbd)):
          if predictedbd[i] == 0:
            predictedbd[i] = 0
          elif predictedbd[i] == 1:
            predictedbd[i] = 3
        confidencebd = get_conf(outputbd)

        _, predictedbh = torch.max(outputbh.data, 1)
        predictedbh = predictedbh.tolist()
        for i in range(len(predicted)):
          if predictedbh[i] == 0:
            predictedbh[i] = 0
          elif predictedbh[i] == 1:
            predictedbh[i] = 4
        confidencebh = get_conf(outputbh)

        _, predictedclco = torch.max(outputclco.data, 1)
        predictedclco = predictedclco.tolist()
        for i in range(len(predicted)):
          if predictedclco[i] == 0:
            predictedclco[i] = 1
          elif predictedclco[i] == 1:
            predictedclco[i] = 2
        confidenceclco = get_conf(outputclco)

        _, predictedcld = torch.max(outputcld.data, 1)
        predictedcld = predictedcld.tolist()
        for i in range(len(predicted)):
          if predictedcld[i] == 0:
            predictedcld[i] = 1
          elif predictedcld[i] == 1:
            predictedcld[i] = 3
        confidencecld = get_conf(outputcld)

        _, predictedclh = torch.max(outputclh.data, 1)
        predictedclh = predictedclh.tolist()
        for i in range(len(predicted)):
          if predictedclh[i] == 0:
            predictedclh[i] = 1
          elif predictedclh[i] == 1:
            predictedclh[i] = 4
        confidenceclh = get_conf(outputclh)

        _, predictedcod = torch.max(outputcod.data, 1)
        predictedcod = predictedcod.tolist()
        for i in range(len(predicted)):
          if predictedcod[i] == 0:
            predictedcod[i] = 2
          elif predictedcod[i] == 1:
            predictedcod[i] = 3
        confidencecod = get_conf(outputcod)

        _, predictedcoh = torch.max(outputcoh.data, 1)
        predictedcoh = predictedcoh.tolist()
        for i in range(len(predicted)):
          if predictedcoh[i] == 0:
            predictedcoh[i] = 2
          elif predictedcoh[i] == 1:
            predictedcoh[i] = 4
        confidencecoh = get_conf(outputcoh)

        _, predicteddh = torch.max(outputdh.data, 1)
        predicteddh = predicteddh.tolist()
        for i in range(len(predicted)):
          if predicteddh[i] == 0:
            predicteddh[i] = 3
          elif predicteddh[i] == 1:
            predicteddh[i] = 4
        confidencedh = get_conf(outputdh)

        total_pred = [predictedbcl, predictedbco, predictedbd,
                      predictedbh, predictedclco, predictedcld, predictedclh,
                      predictedcod, predictedcoh , predicteddh]
        total_conf = [confidencebcl, confidencebco, confidencebd,
                      confidencebh, confidenceclco,confidencecld, confidenceclh,
                      confidencecod, confidencecoh, confidencedh]

        for i in range(10):
          temp = []
          for j in range(batch_size):
            temp.append(total_pred[j][i])
          bi_pred.append(max(temp,key=temp.count))

        total_bi_pred = bi_pred
        #predicted = max(set(total_pred), key=total_pred.count)
        n_samples += labels.size(0)

        list_mean_bi = reduce(lambda x, y: x + y, total_conf)/len(total_conf)

        if confidence < 0.99:
          final_pred.append(total_bi_pred)
        else:
          final_pred.append(predicted)

        n_class_correct = [1,2,3,4,5]
        bi_class_correct = [1,2,3,4,5]
        fin_class_correct = [1,2,3,4,5]
        labels = labels.tolist()
        for i in range(batch_size):
            if predicted[i] == labels[i]:
              n_correct += 1
            if total_bi_pred[i] == labels[i]:
              bi_correct += 1
            if final_pred[0][i] == labels[i]:
              fin_correct += 1
            label = labels[i]
            pred = predicted[i]
            bipred = total_bi_pred[i]
            print("final_pred =", final_pred)
            fin = final_pred[0][i]
            predicted_list.append(pred)
            bipredicted_list.append(bipred)
            final_list.append(fin)
            true_list.append(label)
            if (label == pred):
                n_class_correct[label] += 1
            if (label == bipred):
                bi_class_correct[label] += 1
            if (label == fin):
                fin_class_correct[label] += 1
            n_class_samples[label] += 1


        print("conf",confidence)
    print("n_correct =", n_correct)
    print("5g predictions", predicted_list)
    print("OvO predictions", bipredicted_list)
    print("Final predictions", final_list)

    acc = 100.0 * n_correct / n_samples
    biacc = 100.0 * bi_correct / n_samples
    finacc = 100.0 * fin_correct / n_samples
    print('Accuracy of the network on the 100 test images from 5 Genre: {} %' .format(acc))
    print('Accuracy of the network on the 100 test images from OvO: {} %' .format(biacc))
    print('Accuracy of the combined network on the 100 test images: {} %' .format(finacc))

    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print('Accuracy of {}: {} %' .format(classes[i], acc))

    for i in range(num_classes):
        biacc = 100.0 * bi_class_correct[i] / n_class_samples[i]
        print('Accuracy of {}: {} %' .format(classes[i], biacc))

    for i in range(num_classes):
        finacc = 100.0 * fin_class_correct[i] / n_class_samples[i]
        print('Accuracy of {}: {} %' .format(classes[i], finacc))

    for i in range(len(true_list)):
      true_list[i] = true_list[i]

    f1_score1 = f1_score(true_list, predicted_list, average='weighted')
    print('F1-Score is {}' .format(f1_score1))

    f1_score_bi = f1_score(true_list, bipredicted_list, average='weighted')
    print('F1-Score is {}' .format(f1_score_bi))

    f1_score_fin = f1_score(true_list, final_list, average='weighted')
    print('F1-Score is {}' .format(f1_score_fin))

##############################################
###CONFUSION MATRIX
##############################################

from sklearn import metrics
conf_matrix = metrics.confusion_matrix(true_list, predicted_list)

sns.heatmap(conf_matrix,
            annot=True,
            fmt='g',
            xticklabels=["blues", "classical", "hiphop", "metal", "pop"],
            yticklabels=["blues", "classical", "hiphop", "metal", "pop"])
plt.ylabel('Actual',fontsize=13)
plt.xlabel('Prediction',fontsize=13)
plt.title('Confusion Matrix from OvR',fontsize=17)
plt.show()

conf_matrix_bi = metrics.confusion_matrix(true_list, bipredicted_list)

sns.heatmap(conf_matrix_bi,
            annot=True,
            fmt='g',
            xticklabels=["blues", "classical", "hiphop", "metal", "pop"],
            yticklabels=["blues", "classical", "hiphop", "metal", "pop"])
plt.ylabel('Actual',fontsize=13)
plt.xlabel('Prediction',fontsize=13)
plt.title('Confusion Matrix from OvO',fontsize=17)
plt.show()

conf_matrix_final = metrics.confusion_matrix(true_list, final_list)

sns.heatmap(conf_matrix_final,
            annot=True,
            fmt='g',
            xticklabels=["blues", "classical", "hiphop", "metal", "pop"],
            yticklabels=["blues", "classical", "hiphop", "metal", "pop"])
plt.ylabel('Actual',fontsize=13)
plt.xlabel('Prediction',fontsize=13)
plt.title('Confusion Matrix Final',fontsize=17)
plt.show()

print("Finished Training at: ", datetime.utcnow())

print("Training took ", datetime.utcnow()-start_time)

print('rotors are good sir')