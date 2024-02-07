import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self.assign_weights(np.zeros(n_features))

        for _ in range(0, self.max_iter):
            gradient_sum = 0
            for i in range(n_samples):
                gradient_sum += self._gradient(X[i], y[i])
            gradient = gradient_sum/n_samples
            self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))

        for _ in range(0, self.max_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                gradient_sum = np.zeros_like(self.W)
                for j in range(batch_size):
                    gradient_sum += self._gradient(X_batch[j], y_batch[j])
                gradient = gradient_sum/batch_size
                self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))

        for p in range(0, self.max_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(len(X_shuffled)):
                gradient = self._gradient(X_shuffled[i], y_shuffled[i])
                self.W -= self.learning_rate * gradient

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE

        # gradient of E(w) = -(_y*_x)/(1 + e^(_y*wT*_x))
        num = _y * _x
        dinom = 1 + np.exp(_y * np.dot(self.W.T, _x))
        return -num/dinom

		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        proba = []
        for sample in X:
            pred = 1/(1 + np.exp(- np.dot(self.W.T, sample)))
            proba.append([pred, 1-pred])

        return proba
		### END YOUR CODE

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        probas = self.predict_proba(X)
        y_pred = [1 if proba[0] >= 0.5 else -1 for proba in probas]
        return y_pred
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

