#train image classfication model
import torch
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
import matplotlib.pyplot as plt

#1. DATA PRE-PREPROTRATION 資料預處理

#transforms.Compose裡面包含了所有要對圖片預處理的動作
transform = transforms.Compose(  
    [transforms.ToTensor(), 
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #mean, std
batch_size = 8
train_data = CIFAR10(root='./nn/data',train=True,download=True,transform=transform)
test_data = CIFAR10(root='./nn/data',train=True,download=False,transform=transform)
train_dataLoader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True) #num_workers為loaddata的thread數量
test_dataLoader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)
print("shape=",train_data[0][0].shape)


#2. build model
class IMGMODEL(Module):
    def __init__(self, in_channel, num_class):
        super(IMGMODEL,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, 16, 3, stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(16, 128, 3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.activation = torch.nn.functional.relu
        self.linear1 = torch.nn.Linear(128*8*8, 128)
        self.linear2 = torch.nn.Linear(in_features=128,out_features=num_class)
    
    def forward(self,x):
        x = self.activation(self.pool(self.conv1(x)))
        x = self.activation(self.pool(self.conv2(x)))
        x = x.view(x.shape[0],-1) #batch size = 8
        x = self.linear1(x)
        x = self.linear2(x)
        #x = torch.nn.functional.log_softmax(x)  --> no need because crossentropy already calculate softmax
        
        return x

#3. training
from tqdm.auto import tqdm
Epoch = 5
model = IMGMODEL(3,10)
device = torch.device("cpu")
criteria = torch.nn.CrossEntropyLoss()
optm = torch.optim.SGD(model.parameters(),lr=0.02)
model.train()
for epoch in range(Epoch):
    losses = 0
    for batch in tqdm(train_dataLoader, desc="training..."):
        x,y = batch[0].to(device), batch[1].to(device) #將xy都放進device，單位為batchsize
        predict = model(x) 
        loss = criteria(predict,y)
        optm.zero_grad()
        loss.backward()
        optm.step()
        losses += loss.item()
    print(f"epoch {epoch+1}, loss = {losses}")


#4. evaluate
model.eval()
pre = []
for batch in tqdm(test_dataLoader,desc="testing"):
    x, y = batch[0].to(device), batch[1].to(device)
    predict = model(x)
    ans = torch.argmax(predict, dim=1) #dim?
    for i in range(len(y)):
        pre.append(bool(y[i]==ans[i]))

print(f"accuracy : {pre.count(True)/len(pre)}")


#test sample
img = test_data[6][0]
ans = test_data[6][1]
img = img.view(1,3,32,32) #input shape = (1, 3, 32, 32)
predict = model(img) #model output shape = (1, 10)
predict  = torch.argmax(predict, dim=1)
print(f"predicted : {int(predict)}, actual : {ans}" )



