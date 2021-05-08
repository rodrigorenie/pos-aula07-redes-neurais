import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")

x_np = np.array(list(range(1, 11)))
y_np = x_np ** 2

x = torch.from_numpy(x_np.astype(np.float32)).to(device)
y = torch.from_numpy(y_np.astype(np.float32)).to(device)

y = y.view(y.shape[0], 1)
x = x.view(x.shape[0], 1)

# Criar modelo
n_sample, n_features = x.shape

# model = nn.Linear(n_features, n_features)
model = nn.Sequential(nn.Linear(1, 10),
                      nn.ReLU(),
                      # nn.Sigmoid(),
                      nn.Linear(10, 1))

if use_cuda:
    model.cuda()

# Função de erro
criterion = nn.MSELoss()
criterion = nn.L1Loss()

# Otimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

start = time.time()
epochs = 100000
for e in range(epochs):
    # forward and loss
    y_predict = model(x)
    loss = criterion(y_predict, y)
    # backward
    loss.backward()
    # weights update and zero grad
    optimizer.step()
    optimizer.zero_grad()
    # if (e + 1) % 10 == 0:
    #     print(f'epoch: {e + 1} loss: {loss.item():.4f}')

print(f'tempo: {time.time()-start}')

print(loss.item())
predicted = model(x).to('cpu').detach().numpy()
plt.plot(x_np, y_np, 'ro')
plt.plot(x_np, predicted, 'b')
plt.show()




