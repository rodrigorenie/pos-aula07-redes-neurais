import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


"""
Created on Mon Apr 19 11:32:34 2021

@author: jasmi
"""
"""
1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4) Criar o loop de treinamento
   - forward pass and loss: calcular a predição e o erro
   - backward pass: calcular os gradientes e zerar para o próximo loop
   - update weights: ajuste dos pesos do modelo
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# preparar dados (pq make_regression? pq não aletatório com torch.randn?)
X_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20,
                                      random_state=1)

X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))

# converte y para o mesmo formato de x: uma lista com uma lista de 100 elementos
y = y.view(y.shape[0], 1)

# Criar modelo
n_sample, n_features = X.shape

# Pq nao randn? Pq para um modelo linear deve ser normalizado
model = nn.Linear(n_features, n_features)
if use_cuda:
    model.cuda()

# Função de erro
criterion = nn.MSELoss()

# Otimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Treinar modelo
epochs = 200

for e in range(epochs):
    # forward and loss
    y_predict = model(X)
    loss = criterion(y_predict, y)
    # backward
    loss.backward()
    # weights update and zero grad
    optimizer.step()
    optimizer.zero_grad()
    if (e + 1) % 10 == 0: print(f'epoch: {e + 1} loss: {loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_np, y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()
