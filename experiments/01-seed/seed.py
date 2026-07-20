import numpy as np

#Training data
X = np.array([[0,0],
	      [0,1],
              [1,0],
              [1,1]], dtype=float)

Y = np.array([[0],
             [1],
             [1],
             [1]], dtype=float)

np.random.seed(1)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.5

for epoch in range(10000):
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    error = output - Y
    loss  = np.mean(error ** 2)
    
    output_delta = error * output * (1 - output)
    hidden_delta = (output_delta @ W2.T) * hidden * (1 - hidden)

    W2 -= learning_rate * hidden.T @ output_delta
    b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    W1 -= learning_rate * X.T @ hidden_delta
    b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

print("\nTraining complete")
print("Final loss:", loss)

print("\nPredictions:")
print(output)

np.savez("tiny_nn_weights.npz",
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2)

print("\nWeights saved to tiny_nn_weights.npz")
