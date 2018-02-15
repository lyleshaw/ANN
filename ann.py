import sys, random, json
import numpy as np


class CEcost(object):
    def f(a, y):
        return np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))

    def d(a, y):
        return a - y


class ANN(object):
    def __init__(self, sizes):
        self.layer = len(sizes)
        self.sizes = sizes
        self.initializerW()
        self.cost = CEcost

    def initializerW(self):
        self.bs = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.ws = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.bs, self.ws):
            a = sigmoid(np.dot(w, a) + b)
            return a

    def SGD(self, trainingdata, epochs, minbatchsize, eta, lmbda=0.0, testdata=False, monitor_test_cost=False,
            monitor_test_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False, ):
        n = len(trainingdata)
        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []
        n_data = len(testdata)
        for i in range(epochs):
            random.shuffle(trainingdata)
            minbatchsizes=[trainingdata[k:k+minbatchsize] for k in range(0,n,minbatchsize)]
            for minbatchsize in minbatchsizes:
                self.updata_minbtch(minbatchsize,eta,lmbda,len(trainingdata))
                print ('Epoch' + i + 'has completed')
            if monitor_test_cost:
                cost=self.total_cost(trainingdata,lmbda)
                training_cost=append(cost)
                print ('Training Data COST is '+cost)

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            dnb, dnw = self.backprop(x, y)
            nabla_b = [db + nb for db, nb in zip(dnb, nabla_b)]
            nabla_w = [dw + nw for dw, nw in zip(dnw, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * db for b, db in zip(self.biases, nablab)]
        self.weights = [w * (1 - eta / lmbda / n) - eta / len(mini_batch) * dw for w, dw in zip(self.weights, nablaw)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            d = (self.cost).d(activations[-1],y)
            nabla_b[-1] = d
            nabla_w[-1] = np.dot(d, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            d = np.dot(self.weights[1 - l].transpose(), d) * sp
            nabla_b[-l] = d
            nabla_w[-l] = np.dot(d, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
                cost += self.cost.f(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data) + sum(np.linalgnorm(w) + 2 for w in self.weights))
        return cost


def save(self, name):
    data = {"sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]}
    f = open(name, "w")
    json.dump(data, f)
    f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close
    net = ANN(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    e = np.zeros(10, 11)
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

