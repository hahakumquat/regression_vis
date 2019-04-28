import numpy as np
import matplotlib.pyplot as plt

## Dataset ##
# Creates a single-dimensional dataset given a specified generator function.
# Inputs:
# n: number of datapoints
# x_range: range of inputs
# fn: function creating output y
# noise: Gaussian noise applied to output y
class Dataset():
    def __init__(
        self,
        n=100,
        x_range=(-10, 10),
        fn=lambda x: x + 3,
        noise=0
    ):
        self.n = n
        assert(x_range[0] < x_range[1])
        self.x = self.getx(x_range, n)
        self.y = fn(self.x) + np.random.normal(0, noise, len(self.x))
        self.plot_x = self.getx(x_range, 200)
        
    def getx(self, x_range, n):
        return np.array(range(n)) / n * (x_range[1] - x_range[0]) - (x_range[1] - x_range[0]) / 2

class Model():
    def __init__(
            self,
            dataset,
            features=[lambda x: x],
            gamma=0.1
    ):
        self.dataset = dataset        
        self.features = features + [lambda _: 1]
        self.X = np.array([[f(xi) for f in self.features] for xi in dataset.x])
        self.w = np.random.normal(0, 1, len(self.features))
        self.gamma = gamma
        self.losses = []

    def getX(self, x=None):
        if x is None:
            return self.X
        return np.array([[f(xi) for f in self.features] for xi in x])

    def forward(self, x):
        if x is None:
            x = self.dataset.x
        X = self.getX(x)
        return np.matmul(X, self.w)

    def update(self, alpha):
        self.w -= alpha / self.dataset.n * np.matmul(np.transpose(self.X), model.forward(self.dataset.x) - self.dataset.y) - self.gamma * 2 * self.w

    def loss(self, dataset=None):
        if not dataset:
            dataset = self.dataset
        return (np.sum(np.square(self.forward(dataset.x) - dataset.y)) + self.gamma * np.dot(self.w, self.w)) / 2 / len(dataset.y)

    def record_loss(self, dataset):
        if not dataset:
            dataset = self.dataset
        self.losses.append(self.loss())


class Optimizer():
    def __init__(
            self,
            alpha=0.1,
            epsilon=0.01,
            sim=False,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.losses = []
        self.sim = sim

    def train(self, model, sim=False):
        if self.sim:
            plt.figure()
        while len(self.losses) == 0 or not np.abs(model.loss() - fself.losses[-1]) < self.epsilon:
            plt.clf()
            model.record_loss(dataset)
            model.update(self.alpha)
            if self.sim:
                self.plot(model)
        if self.sim:
            plt.show()
        
    def plot(self, model, pause=0.05):
        plt.subplot(211)
        plt.scatter(model.dataset.x, model.dataset.y)
        plt.plot(model.dataset.plot_x, model.forward(model.dataset.plot_x), color="red")
        plt.legend(model.w)
        plt.subplot(212)
        plt.plot(model.losses)

        plt.pause(pause)

        
dataset = Dataset(n=100, x_range=(-1, 1))
model = Model(dataset, features=[lambda x: x], gamma = 0)
optimizer = Optimizer(sim=True)

optimizer.train(model)
