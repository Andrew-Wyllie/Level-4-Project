####5 genre classification

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
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns


start_time = datetime.utcnow()
print("Starting at: ", start_time)

#I dont have a gpu but i included this line if others have one
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################
from google.colab import drive
drive.mount('/content/gdrive')
######################################################

#Hyperparameters
input_size = 432*288*4
num_classes = 5 ##genres to be seperated
learning_rate = 0.01
batch_size = 10
epochs = 20

##retriving data
train_dir = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/train5'
test_dir = r'/content/gdrive/My Drive/Colab Notebooks/genre/data/images_original/test5'

train_list = glob.glob(os.path.join(train_dir, '*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))

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
        torchvision.transforms.functional.crop(img=img, top=35, left=54, height=218, width=336),

        label = img_path.split('/')[-1][:-9]
        if label == 'blues':
            label = 0
        elif label == 'classical':
            label = 1
        elif label == 'country':
            label = 2
        elif label == 'disco':
            label = 3
        elif label == 'hiphop':
            label = 4


        return img_transformed,label

train_data = dataset(train_list, transform = train_transforms)
test_data = dataset(test_list, transform = test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

classes = ["blues", "classical", "country", "disco", "hiphop"]

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
            nn.Linear(input_size, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

class TestFNN(nn.Module):
    def __init__(self, input_size, num_classes): #dataset [432x288]
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.flatten = nn.Flatten()
        #tensor size is (432, 288)
        self.conv1 = nn.Conv2d(4, 6, 2) ##could use a smaller filter to increase accuracy
        #tensor size is (420, 285)
        self.pool = nn.MaxPool2d(2,2)
        #tensor size is (214, 142)
        self.conv2 = nn.Conv2d(6, 16, 2)
        #tensor size is (210, 138)
        #Max pooling =  (105, 69)

        self.fc1 = nn.Linear(121552, 1024) #115920 = 16*69*105 = channels*height*width => input_size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 5) # 5 => num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x) #-1 defines the size for me
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x) # no soft maax because its include in the loss
        return x


model = FNN(input_size=input_size, num_classes=num_classes).to(device)
model.train()
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

    print("Epoch , {}/{}, Training Accuracy: {}".format(epoch+1, epochs, epoch_accuracy))

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))

    print("Finished with epoch: ", epoch+1)

    loss_diff = abs(last_loss - loss)
    print("Loss difference: ", loss_diff.item())
    if ((loss_diff) < 0.1) & (epoch > 5):
        break
    last_loss = loss

print('Finished Training')

predicted_list = []
true_list = []
model.eval()

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
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            predicted_list.append(pred)
            true_list.append(label)
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print('Accuracy of the network on the 100 test images: {} %' .format(acc))

    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print('Accuracy of {}: {} %' .format(classes[i], acc))



    for i in range(len(true_list)):
      true_list[i] = true_list[i].tolist()
      predicted_list[i] = predicted_list[i].tolist()
    print("Predicted Output: ", predicted_list)
    print("True Output: ", true_list)
    f1_score = f1_score(true_list, predicted_list, average='weighted')
    print('F1-Score is {}' .format(f1_score))

    from sklearn import metrics
    conf_matrix = metrics.confusion_matrix(true_list, predicted_list)
    print(conf_matrix)
###################################################################### 
#Plot the confusion matrix.
sns.heatmap(conf_matrix, 
            annot=True,
            fmt='g', 
            xticklabels=["blues", "classical", "country", "disco", "hiphop"],
            yticklabels=["blues", "classical", "country", "disco", "hiphop"])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()
  
 ######################################################################

#idx = list(map(lambda x: x[0],predicted_list))
#prob = list(map(lambda x: x[1],predicted_list))

#for x in range(len(prob)):
 #   print("x =>", prob[x])
  #  if prob[x] > 0.5:
   #     if prob[x] > 1.5:
    #        if prob[x] > 2.5:
     #           if prob[x] > 3.5:
      #              prob[x] = 'hiphop'
       #         else:
        #            prob[x] = 'disco'
         #   else:
          #      prob[x] = 'country'
#        else:
 #           prob[x] = 'classical'
  #  else:
   #     prob[x] = 'blues'

#submission = pd.DataFrame({'id':idx,'label':prob})

#print(submission)
####################################################################
#fig, axes = plt.subplots(1, 1, figsize=(12, 5))

#print("Loss Output: ", loss_list)
#print("Predicted Output: ", accuracy_list)

#for i in range(len(true_list)):
 # loss_list[i] = loss_list[i].tolist()
  #accuracy_list[i] = accuracy_list[i].tolist()

#plt.subplot(121)
#plt.title("Conv1 + 5LL Model")
#plt.plot(epochs, loss_list)
#plt.grid()
#plt.ylabel('Loss')
#plt.xlabel('Epochs')#

#plt.subplot(122)
#plt.title("Conv1 + 5LL Model")
#plt.plot(epochs, accuracy_list)
#plt.grid()
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.axis((0.8,len(epochs)+0.2,0,1))

#plt.show()
##############################################################################
print("Finished Training at: ", datetime.utcnow())

print("Training took ", datetime.utcnow()-start_time)

print('rotors are good sir')