import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron class with scalar-safe output
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def forward(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total_input).item()  # Return as scalar

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                output = self.forward(X[i])
                error = y[i] - output
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error

# AND and OR logic datasets
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Create and train perceptrons
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X_and, y_and, epochs=1000, learning_rate=0.1)

perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_and, y_or, epochs=1000, learning_rate=0.1)

# Test AND predictions
print("AND Function Predictions:")
for i in range(X_and.shape[0]):
    output = perceptron_and.forward(X_and[i])
    print(f"Input: {X_and[i]} - Predicted Output: {round(output)}")

# Test OR predictions
print("\nOR Function Predictions:")
for i in range(X_and.shape[0]):
    output = perceptron_or.forward(X_and[i])
    print(f"Input: {X_and[i]} - Predicted Output: {round(output)}")
