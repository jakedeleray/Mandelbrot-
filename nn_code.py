import numpy as np 
import pandas as pd  # just using to load in the dataset
from sklearn.model_selection import train_test_split # using for spliting data into training/testing sets
import matplotlib.pyplot as plt

df = pd.read_excel("real_estate.xlsx")
y = df[["Y house price of unit area"]]
X = df.drop(["No", "Y house price of unit area"], axis = 1)


X = X.values    # changing from pandas df to numpy array
y = y.values   # changing from pandas df to numpy array



# setting random seed for reproducibility
np.random.seed(234)

X = (X - X.mean(axis=0)) / X.std(axis = 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 30) # splitting data

class regNN:
    def __init__(self, input_size, hidden1_size, hidden2_size):  # output size will always be one for regression

    
        self.W1 = np.random.randn(input_size, hidden1_size)
        self.B1 = np.zeros((1, hidden1_size))

        self.W2 = np.random.randn(hidden1_size, hidden2_size)
        self.B2 = np.zeros((1, hidden2_size))

        self.W3 = np.random.randn(hidden2_size, 1)
        self.B3 = np.zeros((1, 1))  

    # ReLu activation function works better for regression
    def ReLu(self, x):
        return np.maximum(0,x)
    
    

    def forward_prop(self, X):
        self.Z1 = X @ self.W1 + self.B1
        self.A1 = self.ReLu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.B2
        self.A2 = self.ReLu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.B3  
        self.output = self.Z3       # linear output for regression, so just using identity function

        return self.output

    def back_prop(self, X, y, learning_rate):
        m = X.shape[0]  # number of samples

        # output layer gradient (MSE loss)
        dZ3 = (self.output - y) / m
        dW3 = self.A2.T @ dZ3
        dB3 = np.sum(dZ3, axis=0, keepdims=True)

        # layer 2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (self.Z2 > 0)  # relu derivative
        dW2 = self.A1.T @ dZ2
        dB2 = np.sum(dZ2, axis=0, keepdims=True)

        # layer 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)  # relu derivative
        dW1 = X.T @ dZ1
        dB1 = np.sum(dZ1, axis=0, keepdims=True)

        # gradient descent update
        self.W3 -= learning_rate * dW3
        self.B3 -= learning_rate * dB3

        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * dB2

        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * dB1

        # Compute loss
        loss = np.mean((y - self.output) ** 2) / 2
        return loss
    


    def train(self, X, y, epochs=100, learning_rate=0.01, verbose=True):
        for epoch in range(epochs):
            self.forward_prop(X)
            loss = self.back_prop(X, y, learning_rate)
            rmse = np.sqrt(np.mean((y - self.output) ** 2))
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, RMSE: {rmse:.6f}")


    def predict(self, X):
        self.forward_prop(X)
        return self.output




nn = regNN(6, 5, 5)

nn.train(X_train, y_train, epochs = 2600, learning_rate = 0.01)
# training RMSE is 12.806

y_pred = nn.predict(X_test)

plt.figure(figsize=(6,6))

# scatter plot
plt.scatter(y_test, y_pred, alpha=0.7)

# perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Real vs Predicted House Prices")

plt.grid(True)
plt.show()

# # computing RMSE
# rmse_test = np.sqrt(np.mean((y_test - y_pred) ** 2))

# print(f"Test RMSE: {rmse_test:.6f}")
# # testing RMSE was 16.887
# # once the predictors were normalized, the test RMSE lowed to 11.415


p = nn.predict(X_test[6])
print(p)
