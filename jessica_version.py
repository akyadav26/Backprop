
################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    #print(img[1,:])
    nor = [np.linalg.norm(img[i,:]) for i in range(len(img))]
    nor = np.asarray(nor)
    nor = np.reshape(nor, (len(img), 1))
    img = np.divide(img, nor)
    #print(img[1,:])
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    num_labels = labels.shape[0]
    encoded = np.zeros((num_labels, num_classes))
    encoded[np.arange(num_labels), labels[np.arange(num_labels)]] = 1
    
    return encoded


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    max_val = np.amax(x)
    numerator = np.exp(x-max_val)
    denominator = np.sum(numerator, axis=1).reshape((numerator.shape[0],1))
    return numerator/denominator

    #raise NotImplementedError("Softmax not implemented")


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.
    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        self.x = a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        y = 1/(1 + np.exp((-x))
        return y    
        raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)
        raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        y = x
        y[y<=0] = 0
        return y
        raise NotImplementedError("ReLu not implemented")

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return sigmoid(self.x)*(sigmoid(-self.x))
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return (1-np.tanh(self.x)*np.tanh(self.x))
        raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        y = self.x
        y[y<=0] = 0
        y[y>0] = 1
        return y
        raise NotImplementedError("ReLU gradient not implemented")


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        #self.w = None    # Declare the Weight matrix
        #self.b = None    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.w = np.random.rand(in_units, out_units)
        self.b = np.random.rand(1, out_units)

    def __call__(self, x):
        """
        Make layer callable.
        """
        self.x = x
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        X = np.concatenate((np.ones((x.shape[0],1)), x), axis = 1)
        W = np.concatenate((self.b, self.w), axis = 0)
        self.a = np.dot(X,W)
        return self.a
        raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = np.dot(delta, np.transpose(self.w))
        self.d_w = np.dot(np.transpose(self.x), delta)
        # self.d_b = ??????
        return self.d_x

        raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.
    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        self.x = x
        self.targets = targets
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        lay_one = self.layers[0]
        a_hid_one = lay_one(x)
        z_hid_one = self.layers[1](a_hid_one)

        lay_two = self.layers[2]
        a_hid_two = lay_two(z_hid_one)
        z_hid_two = self.layers[3](a_hid_two)

        fin_lay = self.layers[4]
        a_fin = fin_lay(z_hid_two)
        self.y = softmax(a_fin)
        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        deltas = self.targets - self.y
        delta_secondterm = self.layers[4].backward(deltas)

        deltas = self.layers[3].backward(delta_secondterm)
        delta_secondterm = self.layers[2].backward(deltas)

        deltas = self.layers[1].backward(delta_secondterm)
        delta_secondterm = self.layers[0].backward(deltas)
        
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    epochs = 50
    for i in range(1, epochs+1):
        for j in range(len(x_train), 200):
            model(x_train[j:j+200, :], y_train[j:j+200, :])


    
    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
