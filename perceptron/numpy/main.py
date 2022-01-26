import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Preparing dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
y = y.astype(int)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")


# Shuffling dataset
shuff_idx = np.arange(y.shape[0])
shuff_gen = np.random.RandomState(50)
shuff_gen.shuffle(shuff_idx)

X, y = X[shuff_idx], y[shuff_idx]

X_train, X_test = X[shuff_idx[:70]], X[shuff_idx[70:]]
y_train, y_test = y[shuff_idx[:70]], y[shuff_idx[70:]]

# Normalize
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

class Perceptron():
	def __init__(self, num_features=2, epochs=50):
		self.num_features = num_features
		self.epochs = epochs
		self.weights = np.zeros((num_features, 1), dtype=float)
		self.bias = np.zeros(1, dtype=float)

	def plotter(self, X, y):
		x0_min = np.amin(X[:1])
		x1_min = ((-(x0_min*self.weight[0])-self.bias[0]) / self.weight[1]) 

		x0_max = np.amax(X[:1])
		x1_max = ((-(x0_max*self.weight[0])-self.bias[0] / self.weight[1]) 

	
			
		plt.plot([x0_min, x0_max], [x1_min, x1_max])		
		plt.scatter(X[y==0, 0], X[y==0, 1], label="Class A")
		plt.scatter(X[y==1, 0], X[y==1, 1], label="Class A")
		plt.title("Dataset")
		plt.xlabel("Feature x1")
		plt.ylabel("Feature x2")
		plt.legend()
		plt.show()

	def forward(self, X):
		linear = np.dot(X, self.weights) + self.bias
		predictions = np.where(linear > 0., 1, 0)
		return predictions

	def backward(self, X, y):
		prediction = self.forward(X)
		error = y - prediction
		return error

	def train(self, X, y):
		for e in range(self.epochs):
			for i in range(y.shape[0]):
				errors = self.backward(X[i].reshape(1, self.num_features), y[i]).reshape(-1)
				# Update weights
				self.weights += (X[i]*errors).reshape(self.num_features, 1)
				self.bias += errors

	def evaluate(self, X, y):
		predictions = self.forward(X).reshape(-1)
		# Calculating accuracy -> sum where predictions are equal to real values, then divided by the amount of them
		accuracy = np.sum(predictions==y) / y.shape[0]

perceptron = Perceptron(2, 50)
perceptron.train(X_train, y_train)
perceptron.evaluate(X_test, y_test)
perceptron.plotter(X_train, y_train)
