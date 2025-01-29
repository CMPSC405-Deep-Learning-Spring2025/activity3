import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, input_size, lr=0.01):
        """
        Initialize the Linear Regression model.

        Parameters:
        input_size (int): Number of input features.
        lr (float): Learning rate for gradient descent.
        """
        self.weights = np.zeros(input_size + 1)  # Initialize weights (including bias) to zeros
        self.lr = lr  # Set learning rate

    def predict(self, x):
        """
        Predict the output for a given input x.

        Parameters:
        x (numpy array): Input features.

        Returns:
        float: Predicted output.
        """
        return self.weights.T.dot(np.insert(x, 0, 1))  # Insert bias term and compute dot product

    def fit(self, X, y, epochs=1000, loss_function='mse'):
        """
        Train the Linear Regression model using gradient descent.

        Parameters:
        X (numpy array): Training input features.
        y (numpy array): Training target values.
        epochs (int): Number of iterations over the training data.
        loss_function (str): Loss function to use ('mse', 'mae', 'huber').
        """
        for _ in range(epochs):
            for i in range(y.shape[0]):
                x = np.insert(X[i], 0, 1)  # Insert bias term
                y_pred = self.predict(X[i])  # Predict the output
                error = y[i] - y_pred  # Calculate the error
                if loss_function == 'mse':
                    gradient = error  # Gradient for MSE
                elif loss_function == 'mae':
                    gradient = np.sign(error)  # Gradient for MAE
                elif loss_function == 'huber':
                    delta = 1.0
                    if abs(error) <= delta:
                        gradient = error  # Gradient for small error in Huber loss
                    else:
                        gradient = delta * np.sign(error)  # Gradient for large error in Huber loss
                self.weights += self.lr * gradient * x  # Update weights using gradient descent

    def mse_loss(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE) loss.

        Parameters:
        y_true (numpy array): True target values.
        y_pred (numpy array): Predicted target values.

        Returns:
        float: MSE loss.
        """
        #TODO: implement the MSE function

    def mae_loss(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE) loss.

        Parameters:
        y_true (numpy array): True target values.
        y_pred (numpy array): Predicted target values.

        Returns:
        float: MAE loss.
        """
        #TODO: implement MAE function

    def huber_loss(self, y_true, y_pred, delta=1.0):
        """
        Calculate Huber loss.

        Parameters:
        y_true (numpy array): True target values.
        y_pred (numpy array): Predicted target values.
        delta (float): Threshold parameter for Huber loss.

        Returns:
        float: Huber loss.
        """
        #TODO: implement huber function

    def plot_data_distribution(self, X, y):
        """
        Plot the distribution of the dataset features.

        Parameters:
        X (numpy array): Input features.
        y (numpy array): Target values.
        """
        plt.figure(figsize=(10, 6))
        for i in range(X.shape[1]):
            plt.subplot(3, 4, i + 1)
            plt.hist(X[:, i], bins=20, alpha=0.7)
            plt.title(f'Feature {i+1}')
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, X_train, y_train, X_val, y_val, epochs=1000, loss_function='mse'):
        """
        Plot the learning curve to evaluate overfitting and underfitting.

        Parameters:
        X_train (numpy array): Training input features.
        y_train (numpy array): Training target values.
        X_val (numpy array): Validation input features.
        y_val (numpy array): Validation target values.
        epochs (int): Number of iterations over the training data.
        loss_function (str): Loss function to use ('mse', 'mae', 'huber').
        """
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.fit(X_train, y_train, epochs=1, loss_function=loss_function)
            y_train_pred = np.array([self.predict(x) for x in X_train])
            y_val_pred = np.array([self.predict(x) for x in X_val])
            if loss_function == 'mse':
                train_loss = self.mse_loss(y_train, y_train_pred)
                val_loss = self.mse_loss(y_val, y_val_pred)
            elif loss_function == 'mae':
                train_loss = self.mae_loss(y_train, y_train_pred)
                val_loss = self.mae_loss(y_val, y_val_pred)
            elif loss_function == 'huber':
                train_loss = self.huber_loss(y_train, y_train_pred)
                val_loss = self.huber_loss(y_val, y_val_pred)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), train_losses, label='Training Loss')
        plt.plot(range(epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curve')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load real dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(input_size=X.shape[1])

    # Visualize data distribution
    model.plot_data_distribution(X, y)

    # Train and evaluate with different loss functions
    for loss_function in ['mse', 'mae', 'huber']:
        model.fit(X_train, y_train, epochs=1000, loss_function=loss_function)
        y_pred = np.array([model.predict(x) for x in X_test])
        if loss_function == 'mse':
            loss = model.mse_loss(y_test, y_pred)
        elif loss_function == 'mae':
            loss = model.mae_loss(y_test, y_pred)
        elif loss_function == 'huber':
            loss = model.huber_loss(y_test, y_pred)
        print(f'Loss function: {loss_function}, Loss: {loss}')

    # Plot learning curve to evaluate overfitting and underfitting
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.plot_learning_curve(X_train, y_train, X_val, y_val, epochs=1000, loss_function='mse')