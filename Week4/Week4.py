import numpy as np
import pandas as pd
class LogisticRegression:
    def __init__(self, input_size, hidden_layer_size, learning_rate=0.01, epochs=10000):

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1 = np.random.randn(input_size, hidden_layer_size) * 0.01
        self.b1 = np.zeros((1, hidden_layer_size))
        self.W2 = np.random.randn(hidden_layer_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.tanh(Z1)  
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        cache = {
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2
        }
        return A2, cache
    
    def compute_cost(self, A2, Y):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        cost = np.squeeze(cost)  
        return cost
    
    def back_propagation(self, X, Y, cache):
        m = X.shape[0]
        A1 = cache['A1']
        A2 = cache['A2']
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.power(A1, 2)) 
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        gradients = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2
        }
        return gradients
    
    def update_parameters(self, gradients):
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def fit(self, X, Y):
        self.costs = [] 
        for i in range(self.epochs):
            A2, cache = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            gradients = self.back_propagation(X, Y, cache)
            self.update_parameters(gradients)
            if i % 100 == 0:
                self.costs.append(cost)
            if i % 1000 == 0:
                print(f"Iteration {i}, Cost: {cost}")
    
    def predict(self, X):
        A2, _ = self.forward_propagation(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions
    

