# import torch
# import numpy as np
# x = torch.randn(4,4)
# print(x.size())
# y = x.view(-1,16)
# print(y.size())
# print(y[0][0].item())
# z = np.ones((4,4))
# print(z)
# z = torch.from_numpy(z)
# print(z)


# Linear regression
import torch 
import numpy as np
import matplotlib.pyplot as plt
# hyperparameters
input_features = 1
output_features = 1
learning_rate = 1e-3
epochs = 60 
# data
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = torch.nn.Linear(input_features, output_features)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
x_train1 = torch.from_numpy(x_train)
y_train1 = torch.from_numpy(y_train)
for epoch in enumerate(range(epochs),1):
    #forward pass
    pred = model(x_train1)
    #calculate loss
    loss = criterion(pred,y_train1)
    #calculate gradient
    loss.backward()
    #update weights
    optimizer.step()
    #zero out gradients
    optimizer.zero_grad()
    if(epoch%10==0):
        print("loss: ",loss.item())

#model.load_state_dict(torch.load('model.ckpt'))
# Plot the graph
final_pred = model(x_train1)
final_pred = final_pred.data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, final_pred, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
