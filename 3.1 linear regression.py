# linera regression @YoCoco2233 9/22/2024
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# make data
x_train = torch.randn(100,1)*10
y_train = x_train + 3 * torch.randn(100,1)

# Check if CUDA Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Move data to available device
x_train = x_train.to(device)
y_train = y_train.to(device)

# definition model structure
class LineraRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LineraRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

# active model and move to devices
model = LineraRegressionModel(input_dim, output_dim).to(device)

# definition loss function and Optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# train model
epochs = 10
for epoch in range(epochs):
    # clear Gradient
    optimizer.zero_grad()

    # front ward
    outputs = model(x_train)

    # calculate loss
    loss = criterion(outputs, y_train)

    # backward and Optimizer
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch[{epoch+1}/{epochs}],Loss:{loss.item()}')

# move data to CPU to make picture
predicted = model(x_train).detach().cpu().numpy()
x_train_cpu = x_train.cpu().numpy()
y_train_cpu = y_train.cpu().numpy()

# Draw Canvas
plt.plot(x_train_cpu, y_train_cpu,'ro',label='Original data')
plt.plot(x_train_cpu,predicted, label='Fitted line')
plt.legend()
plt.show()

