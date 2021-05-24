import pandas
import numpy

from matplotlib import pyplot
from tensorflow import keras

from typing import Optional
from collections.abc import Iterator


class MnistData:
    TrainData = tuple[numpy.array, numpy.array]
    TestData = tuple[numpy.array, numpy.array]

    @staticmethod
    def dataset_split() -> tuple[TrainData, TestData]:
        dataset = keras.datasets.mnist.load_data()
        (train_i, train_l), (test_i, test_l) = dataset

        train_i = train_i.reshape((60000, 28 * 28))
        train_i = train_i.astype('float32') / 255
        train_l = keras.utils.to_categorical(train_l)

        test_i = test_i.reshape((10000, 28 * 28))
        test_i = test_i.astype('float32') / 255
        test_l = keras.utils.to_categorical(test_l)

        return (train_i, train_l), (test_i, test_l)

    (train_i, train_l), (test_i, test_l) = dataset_split.__func__()
    train = (train_i, train_l)
    test = (test_i, test_l)


class NetworkData:

    def __init__(self, epochs: Optional[int] = 10,
                 batch_size: Optional[int] = 128,
                 layers: Optional[list[int]] = None) -> None:

        if layers is None:
            layers = [50, 10]

        if len(layers) < 2:
            raise ValueError('Number of layers must be at least 2')

        if layers[-1] != 10:
            raise ValueError('Last layer must have exactly 10 neurons')

        self._epochs = epochs
        self._batch_size = batch_size
        self._layers = layers
        self._history = None
        self._loss = None
        self._accuracy = None

    @property
    def __iterlayer(self) -> Iterator[keras.layers.Dense]:
        layers = self._layers.copy()

        nf = layers.pop(0)
        nl = layers.pop(-1)

        yield keras.layers.Dense(nf, activation='relu', input_shape=(28 * 28,))

        for ni in layers:
            yield keras.layers.Dense(ni, activation='relu')

        yield keras.layers.Dense(nl, activation='softmax')

    def init(self) -> keras.models.Sequential:
        network = keras.models.Sequential()

        for layer in self.__iterlayer:
            network.add(layer)

        network.compile(optimizer='adam', loss='categorical_crossentropy',
                        metrics=['accuracy'])

        history = network.fit(MnistData.train_i, MnistData.train_l,
                              validation_data=MnistData.test,
                              epochs=self._epochs, batch_size=self._batch_size)

        loss, accuracy = network.evaluate(MnistData.test_i, MnistData.test_l)

        self._history = history
        self._loss = loss
        self._accuracy = accuracy
        return network

    @property
    def df(self) -> pandas.DataFrame:
        df = {'epochs': [self._epochs],
              'batch_size': [self._batch_size],
              'history': [self._history],
              'loss': [self._loss],
              'accuracy': [self._accuracy],
              'layer_in': [self.layer_in],
              'layer_out': [self.layer_out]}

        for i, n in enumerate(self.layer_mid):
            df[f'layer_mid{i+1}'] = n

        return pandas.DataFrame.from_dict(df)

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def layers(self) -> list[int]:
        return self._layers

    @property
    def layer_in(self) -> int:
        return self._layers[0]

    @property
    def layer_mid(self) -> list[int]:
        layer_mid = self._layers.copy()
        layer_mid.pop(0)
        layer_mid.pop(-1)
        return layer_mid

    @property
    def layer_out(self) -> int:
        return self._layers[-1]

    @property
    def history(self):
        return self._history

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def accuracy(self) -> float:
        return self._accuracy


class NetworkGenerator:

    def __init__(self):

        try:
            df = pandas.read_pickle('network.pkl')

        except FileNotFoundError:
            df = pandas.DataFrame()

            epochs = [10, 25, 50]
            batches = [100, 300, 600]
            layers2n = [
                [50, 10], [100, 10],
                [400, 10], [500, 10],
                [800, 10], [900, 10]]
            layers3n = [
                [50, 50, 10], [100, 100, 10],
                [400, 400, 10], [500, 500, 10],
                [800, 800, 10], [900, 900, 10],
            ]

            # epochs = [1, 2]
            # batches = [600]
            # layers2n = [[50, 10], [100, 10]]
            # layers3n = [[50, 50, 10], [100, 100, 10]]

            for layers in layers2n:
                for epoch in epochs:
                    for batch in batches:
                        mn = NetworkData(epochs=epoch, batch_size=batch,
                                         layers=layers)
                        mn.init()
                        df = df.append(mn.df.drop(columns=['history']))

            for layers in layers3n:
                for epoch in epochs:
                    for batch in batches:
                        mn = NetworkData(epochs=epoch, batch_size=batch,
                                         layers=layers)
                        mn.init()
                        df = df.append(mn.df.drop(columns=['history']))

        try:
            df.to_pickle('network.pkl', protocol=-1)

        except IOError:
            pass

        self.df = df.reset_index(drop=True).fillna(0)

    def plot(self) -> None:
        x = numpy.arange(self.df.shape[0])
        g1y1 = self.df['accuracy'].to_numpy() * 100
        g1y2 = self.df['loss'].to_numpy() * 100

        g2y = self.df['epochs'].to_numpy()
        g3y = self.df['batch_size'].to_numpy()

        g4y1 = self.df['layer_in'].to_numpy()
        g4y2 = self.df['layer_mid1'].to_numpy()

        g4: pyplot.Axes
        fig, (g1, g2, g3, g4) = pyplot.subplots(4, 1, sharex='col')

        g1.set_xlabel('Accuracy / Loss')
        g2.set_xlabel('Epochs')
        g3.set_xlabel('Batch Size')
        g4.set_xlabel('Layer 0/1/2')

        g1.plot(x, g1y1, color='b')
        g1.twinx().plot(g1y2, color='r')

        g2.bar(x, g2y, width=0.6, color='r', align='center')
        g3.bar(x, g3y, width=0.6, color='g', align='center')

        g4.bar(x, g4y1, width=0.6, color='r', align='center', bottom=g4y2)
        g4.bar(x, g4y2, width=0.6, color='g', align='center')

        fig.tight_layout()
        pyplot.show()


if __name__ == '__main__':
    gen = NetworkGenerator()
    print(gen.df)
    gen.plot()

