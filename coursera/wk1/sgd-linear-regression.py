import matplotlib.pyplot as plt
import numpy as np

# Set the parameters
n = 100
x = np.arange(n)
y0 = [20] * n

# Our true values
theta_0 = -3
theta_1 = .8

noise = np.random.normal(size=n) + 5
y = theta_0 + theta_1 * x + noise

# y = theta_0 + theta_1 * x
def linear_regression(X, y, theta_1=0, theta_0=0, epochs=1000, learning_rate=0.0001):
     for i in range(epochs):
          y_current = (theta_1 * X) + theta_0
         
          # Compute the MSE
          cost = sum([data**2 for data in (y-y_current)]) / n

          #
          theta_1_grad = -(2/n) * sum(X * (y - y_current))
          theta_0_grad = -(2/n) * sum(y - y_current)
          theta_1 = theta_1 - (learning_rate * theta_1_grad)
          theta_0 = theta_0 - (learning_rate * theta_0_grad)
     return theta_1, theta_0, cost

theta_1, theta_0, cost = linear_regression(x,y, epochs=5)
print("theta_0 is {} and theta_1 is {}".format(theta_0, theta_1))

plt.figure(figsize=(20, 10))
plt.title("Random Series")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y);
plt.scatter(x, theta_1 * x + theta_0, color='red')
plt.legend(['Hypothesis', 'SGD Regression Line'], loc='best')
plt.show()
