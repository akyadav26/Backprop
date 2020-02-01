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
import matplotlib.pyplot as plt


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    nor = np.linalg.norm(img, axis = 1)
    nor = np.reshape(nor, (len(img), 1))
    img = np.divide(img, nor)
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
        y = 1/(1 + np.exp((-x)))
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
        return (y)
        raise NotImplementedError("ReLu not implemented")

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x)*(self.sigmoid(-self.x))
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
        self.d_v_w = None
        self.d_v_b = None

        self.w = np.random.normal(0, np.sqrt(1/in_units), (in_units, out_units))
        self.b = np.zeros((1, out_units))
        self.d_v_w = np.zeros((in_units, out_units))
        self.d_v_b = np.zeros((1, out_units))

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
        self.d_w = np.dot(np.transpose(self.x), delta)/self.x.shape[0]
        self.d_b = np.mean(delta, axis=0).reshape((1, delta.shape[1]))
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
        inputs, i = x, 0
               
        while i < len(self.layers):
            outputs = self.layers[i](inputs)
            i += 1
            if(i==len(self.layers)):
                break
            inputs = self.layers[i](outputs)
            i += 1

        self.y = softmax(outputs)

        return (self.y, self.loss(self.y, targets))
        
        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        #print(logits)
        logits = np.log(logits)
        error = -np.multiply(targets, logits)
        return np.sum(error)/targets.shape[0]
        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        deltas = -(self.targets - self.y)
        i = len(self.layers)-1

        while i>=0:
            prev_delta = self.layers[i].backward(deltas)
            i -= 1
            if(i<0):
                break
            deltas = self.layers[i].backward(prev_delta)
            i -= 1
        return
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    
    epochs = config['epochs']
    threshold = config['early_stop_epoch']
    alpha = config['learning_rate']
#    val_loss = 10000*np.ones((epochs,1))
    beta = config['momentum_gamma']
    batch_size = config['batch_size']
    
    N = x_train.shape[0]    
    num_batches  = int((N+batch_size -1 )/ batch_size)
    
    best_weight = []
    best_epoch  = []
    best_bias   = []
    #print(len(model.layers))
    train_loss_list = []
    
    train_acc_list = []
    val_acc_list   = []
    val_loss_list  = []
    
    counter = 0
    
    lam = 0
    
      
    for i in range(1, epochs+1):
        shuffled_indices = np.random.permutation(range(N))
        
        for batch in range(num_batches):
            minibatch_indices = shuffled_indices[batch_size*batch:min(batch_size*(batch+1), N)]
            #print(len(minibatch_indices))
            xbatch = x_train[minibatch_indices, :]
            ybatch = y_train[minibatch_indices, :]
            #print(ybatch.shape)
            y, loss = model(xbatch, ybatch)
                                
            model.backward()            
            #weight update and storing
            for k in range(0, len(config['layer_specs']), 2):
                mom_w = -model.layers[k].d_v_w * beta + alpha*(model.layers[k].d_w + lam*model.layers[k].w )
                mom_b = -model.layers[k].d_v_b * beta + alpha*(model.layers[k].d_b + lam*model.layers[k].b )
                model.layers[k].w = model.layers[k].w - (mom_w  )
                model.layers[k].b = model.layers[k].b - (mom_b  )
                model.layers[k].d_v_w = -mom_w
                model.layers[k].d_v_b = -mom_b     

        y, loss = model(x_train, y_train)  
        train_loss_list.append(loss)
        
        train_pred = np.argmax(y, axis=1)  
        acc = np.mean(np.argwhere(y_train==1)[:,1]==train_pred) 
        
        train_acc_list.append(acc)
        
        
        print("Training acc for epoch ", i, " is:\n", acc) 
        print("Training loss for epoch ", i, " is:\n", loss) 
        val_y, val_loss = model(x_valid, y_valid)
        val_loss_list.append(val_loss)

        val_pred = np.argmax(val_y, axis=1)  
        acc = np.mean(np.argwhere(y_valid==1)[:,1]==val_pred) 
        val_acc_list.append(acc)
        
        print("Validation acc for epoch ", i, " is:\n", acc) 
        print("Validation loss for epoch ", i, " is:\n", val_loss)
        if(i>1 and val_loss <min(val_loss_list[:-1])):
            #update best weights
            counter = 0
            weight = []
            bias = []
            for k in range(0, len(config['layer_specs']), 2):
                weight.append(model.layers[k].w)
                bias.append(model.layers[k].b)
            best_weight = weight    
            best_bias = bias
            best_epoch = i
        else:
            counter +=1
        
        if counter > threshold:
            print("best epoch:", best_epoch)
            break

#        if(i>=6 and val_loss[i-1]>=val_loss[i-2] and val_loss[i-2]>=val_loss[i-3]and val_loss[i-3]>=val_loss[i-4]and val_loss[i-4]>=val_loss[i-5]and val_loss[i-5]>=val_loss[i-6]):
#            break
    
    #print(len(best_weight))
    #print('Epoch: ', i)
    #print(val_loss)
    p = 0
    for k in range(0, len(config['layer_specs']), 2):
        model.layers[k].w = best_weight[p]
        model.layers[k].b = best_bias[p]
        p = p + 1
    
    return train_loss_list, val_loss_list, train_acc_list, val_acc_list
    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    pred, loss = model(X_test, y_test)
    test_pred = np.argmax(pred, axis=1)  
    acc = np.mean(np.argwhere(y_test==1)[:,1]==test_pred) 

    print("Test acc is:\n", acc) 
    return test
    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x, y = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    #print(x.shape[0])
    # Create splits for validation data here.
    # x_valid, y_valid = ...
    x_train, y_train = x[0:int(0.8*x.shape[0]), :], y[0:int(0.8*y.shape[0]), :]
    x_valid, y_valid = x[int(0.8*x.shape[0]):, :], y[int(0.8*y.shape[0]):, :]
    
    # train the model
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = train(model, x_train, y_train, x_valid, y_valid, config)

#    val_loss_list = val_loss[:len(val_acc_list)]
    
    x = [i for i in range(1, len(train_loss_list) + 1)]

    plt.title("Loss vs. Number of epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.plot(x, train_loss_list, color='r', label='training loss')
    plt.plot(x, val_loss_list, color = 'b', label = 'validation loss')
    
    plt.legend()
    plt.show()
    
    plt.title("Accuracies vs. Number of epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, train_acc_list, color='r', label='training accuracy')
    plt.plot(x, val_acc_list, color = 'b', label='validation accuracy')
    
    plt.legend()
    plt.show()    
    
    
    
    
    test_acc = test(model, x_test, y_test)
