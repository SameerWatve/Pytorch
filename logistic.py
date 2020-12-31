
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Hyper-parameters 
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

class Net(nn.Module):
    def __init__(self,input_features,hidden_features,output_features):
        super(Net,self).__init__()
        self.hidden = nn.Linear(input_features,hidden_features)
        self.output = nn.Linear(hidden_features,output_features)
    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = Net(input_features=input_size,hidden_features=100,output_features=num_classes)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (images,lables) in enumerate(train_dataloader):
        pred = model(images)
        loss = loss_fn(pred,lables)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if(batch_idx%100==0):
            print("epoch {}, batch {} loss {}".format(epoch, batch_idx,loss.item()))

#model.load_state_dict(torch.load('model.ckpt'))
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
pass

    