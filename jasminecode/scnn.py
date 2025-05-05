import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from PIL import Image
import PIL.ImageOps 
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim




# Creating some helper functions
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset):
        print("slay")
        self.imageFolderDataset = imageFolderDataset
        #self.transform = transform
        self.setTransforms()
        #self.getsizes()

    def __getitem__(self, index):
        # image 0 is a random image from the set
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #print("img0 = "+ img0_tuple[0][32:])
        
        #we then split the path to get a list of [class (patient id), file name]
        img0_split = img0_tuple[0].split("/")
        img0_class = img0_split[2].split("\\")
        #print("img0 = ", img0_class)

        #roughly half images should be the same
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            #copy the list to randomise it
            randomlist = self.imageFolderDataset.imgs
            #and shuffle said list
            random.shuffle(randomlist)

            #go through every item in list until one from same class, but a different image is found
            for i in range(self.__len__()):
                #print(randomlist[i])
                #split path name again, same format as image 1
                randomlist_split = randomlist[i][0].split("/")
                img1_class = randomlist_split[2].split("\\")
                #print(img1_class)

                #if patient id is same but its not the same image, that becomes image 1 and we stop looping
                if img1_class[0] == img0_class[0] and img1_class[1]!=img0_class[1]:

                #   if list2[i][0][32:-10] == img0_tuple[0][32:-10] and list2[i][0] != img0_tuple[0]:
                    img1_tuple = randomlist[i]
                    #print("break")
                    break
            
                

                #img1_tuple = random.choice(self.imageFolderDataset.imgs)
            #print("same: img 1 = "+img1_tuple[0][32:])
                #if img0_tuple[0] == img1_tuple[1] or count>10:
                    #break
                #count+=1
        else:
            while True:
                #finding image from different class
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                #print("diff: img 1 = "+img1_tuple[0][32:])
                img1_class = img1_tuple[0].split("/")
                img1_class = img1_class[2].split("\\")

                if img0_class[0] != img1_class[0]:
                    break

        ##open both images
        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple[0]).convert('RGB')
        
        
        
        #L = grayscale, P = is 256 colours, so less space then og ??
        #img0 = img0.convert("P")
        #img1 = img1.convert("P")

        #transform the images, based on the transformation input
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        #self.getsize(img1)
        #width, height = img0.size
        #print("0 = W:",width,"H:",height)   
        #width, height = img1.size
        #print("1 = W:",width,"H:",height) 
        #self.getsize(img0)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    def setTransforms(self):
        self.transform = transforms.Compose([
        transforms.CenterCrop(2000),
        transforms.Resize(size = (300,300)),
        transforms.ToTensor()
        ])

    def getsize(self, img):
        width, height = img.size
        print("W:",width,"H:",height)
        
    def getsizes(self):
        
        for i in self.imageFolderDataset:
            width, height = i.size
            print("W:",width,"H:",height)




"""

# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=8)

print("3\n")
#getting stuck here ::::::
# Extract one batch
example_batch = next(iter(vis_dataloader))
print("4\n")
# Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]),0)
print("6\n")
print(example_batch[2].numpy().reshape(-1))
print("5\n")
imshow(torchvision.utils.make_grid(concatenated))
"""

class SiameseNetwork(nn.Module):
    def __init__(self):
        
        super(SiameseNetwork, self).__init__()
        """
        #convlutional layer, 3 is for rgb, 6 idk why, but is the first in the next layer, 5 is the kernal size 
        self.conv1 = nn.Conv2d(3,6,5)
        #pool typically uses 2,2
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)

        #fully connected layers, are executed after all convs have been executed
        self.fc1 = nn.Linear(3952144,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3,143,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride = 2),

            nn.Conv2d(143, 1206, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride = 2),

            nn.Conv2d(1206, 67, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride = 2)


        )
        """
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        """

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(72963, 3097),
            nn.ReLU(inplace=True),
            
            nn.Linear(3097, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        #x = x.view(x.size(0), -1)
        #print(x+"=numberr")
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2






# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
    

folder_dataset = datasets.ImageFolder(root="picture_all_visits/train - Copy/")
print("1\n")
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset)
print("2\n")


train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)

print("3\n")
#device = torch.device("cpu")
net = SiameseNetwork().cpu()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
print("4\n")
counter = []
loss_history = [] 
iteration_number= 0
if __name__ == '__main__':
    # Iterate throught the epochs
    for epoch in range(10):
        print("5")
        # Iterate over batches
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
        #print(enumerate(train_dataloader)[0])
        #while next(train_dataloader):
        #for data in train_dataloader:
            #print (data[0],"idk",data[1])
            #print(img0[0])
            #print("6")
            
            img0, img1, label = img0.cpu(), img1.cpu(), label.cpu()
            
            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print( "Epoch: ",epoch,"\nCurrent loss:",loss_contrastive.item())
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        print("7")

    #show_plot(counter, loss_history)
        

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_dataset_test = datasets.ImageFolder(root="picture_all_visits/test - Copy/")
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test)
    test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

    # Grab one image that we are going to test

    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    for i in range(10):
        # Iterate over 10 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)
        print(next(dataiter)[2].numpy().reshape(-1))
        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)
        #print(x0)
        #print(x1)
        
        output1, output2 = net(x0.cpu(), x1.cpu())
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        #imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
        print("Dissimilarity: ",euclidean_distance.item())

print("10")