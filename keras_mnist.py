"""
Criado em Wed Mar 31 16:00:00 2021

@author: Jasmine Moreira

1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador
4) Criar o loop de treinamento
   - forward pass: calcular a predição e o erro
   - backward pass: calcular os gradientes
   - update weights: ajuste dos pesos do modelo
"""
import random
from typing import Tuple

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


rs = [range(50, 501), 10]
epochs = 30
batch_size = 512


grid = {
    2: {(l1, 10): 0 for l1 in random.sample(*rs)},
    3: {(l1, l2, 10): 0 for l1, l2 in zip(random.sample(*rs),
                                          random.sample(*rs))}
}






for nlayer, networkinfo in grid.items():
    for layerinfo, result in networkinfo.items():
        if nlayer == 2:
            l1, l2 = layerinfo

            network = models.Sequential()
            network.add(layers.Dense(l1, activation='relu', input_shape=(28 * 28,)))
            network.add(layers.Dense(l2, activation='softmax'))

            network.compile(optimizer='adam', loss='categorical_crossentropy',
                            metrics=['accuracy'])

            history = network.fit(train_images, train_labels, epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(test_images, test_labels))

            test_loss, test_acc = network.evaluate(test_images, test_labels)
            print(f'\t{l1} | {l2} neurônios: {test_acc}')
            grid[nlayer][layerinfo] = test_acc

        if nlayer == 3:
            l1, l2, l3 = layerinfo

            network = models.Sequential()
            network.add(layers.Dense(l1, activation='relu',
                                     input_shape=(28 * 28,)))
            network.add(layers.Dense(l2, activation='relu'))
            network.add(layers.Dense(l3, activation='softmax'))

            network.compile(optimizer='adam', loss='categorical_crossentropy',
                            metrics=['accuracy'])

            history = network.fit(train_images, train_labels, epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(test_images, test_labels))

            test_loss, test_acc = network.evaluate(test_images, test_labels)
            print(f'\t{l1} | {l2} | {l3} neurônios: {test_acc}')
            grid[nlayer][layerinfo] = test_acc

for nlayer, networkinfo in grid.items():
    print(f'{nlayer} camadas:')
    for layerinfo, result in networkinfo.items():
        if nlayer == 2:
            l1, l2 = layerinfo
            print(f'\t{l1} | {l2} neurônios: {result}')

        if nlayer == 3:
            l1, l2, l3 = layerinfo
            print(f'\t{l1} | {l2} | {l3} neurônios: {result}')


raise SystemExit

network = models.Sequential()
network.add(layers.Dense(50, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = network.fit(train_images,
                      train_labels,
                      epochs=10,
                      batch_size=128,
                      validation_data=(test_images, test_labels))

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)

raise SystemExit

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len( history_dict['loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('training_validation_loss.png')

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('training_validation_acc.png')
