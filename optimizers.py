import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
# torch.manual_seed(1)    # reproducible

LR = 0.01
batch_size = 8
EPOCH = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# default network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(784, 20)   # hidden layer
        self.out = torch.nn.Linear(20, 10)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)             # linear output
        return x

if __name__ == '__main__':
    # different nets
    net_SGD         = Net()
    net_Momentum    = Net()
    net_RMSprop     = Net()
    net_Adam        = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # different optimizers
    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    loss_func = torch.nn.CrossEntropyLoss()
    losses_his = [[], [], [], []]   # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(train_dataloader):   
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            b_x = b_x.reshape(-1,784)
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)              # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                l_his.append(loss.data.numpy())     # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

    # testing
    with torch.no_grad():
        correct = 0
        total = 0
        for ind, net in enumerate(nets):
            for images, labels in test_dataloader:
                images = images.reshape(-1, 784)
                outputs = net(images)
                _, predicted = torch.max(outputs,dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Accuracy of the model {} {} on the 10000 test images: {} %'.format(ind, net, 100 * correct / total))