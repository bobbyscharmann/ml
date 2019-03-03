import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Simple parabola
def func_y_concave(x):
    y = x**2 - 4*x + 2
    return y

# Derivative of the func_y defined above
def cost_function_concave(x):
    cost = 2*x - 4
    return cost

# Performs stochastic gradient descent using the above defined cost function
# Liberally stolen from:
# https://medium.com/@rrfd/what-is-a-cost-function-gradient-descent-examples-with-python-16273460d634
def gradient_descent(initial_guess, learning_rate, epoch):
    x_gd = []
    y_gd = []
    
    x_gd.append(initial_guess)
    y_gd.append(func_y_concave(initial_guess))
    previous_x = initial_guess
    for i in range(epoch):
        current_x = 0
        current_y = 0
        current_x = previous_x - learning_rate * (cost_function_concave(previous_x))
        current_y = func_y_concave(current_x)

        x_gd.append(current_x)
        y_gd.append(current_y)

        previous_x = current_x

    return x_gd, y_gd

# Initial guesses to apply for SGD
initial_guesses = [1.2, 2.3]

# Learning rates to apply for SGD
learning_rates = [0.1, 0.2, 0.4]

# Number of epochs to apply for SGD
num_epochs = 50

# Configure the plot figure. Note the subplot for each initial guess and
# learning rate
gs = gridspec.GridSpec(len(initial_guesses), len(learning_rates))
fig = plt.figure(tight_layout=True, figsize=(20, 10))

# Loop over each initial guess and learning rate and run SGD and plot it
for i in range(len(initial_guesses)):
    for j in range(len(learning_rates)):
        # Add this subpot
        ax = fig.add_subplot(gs[i, j])

        # Grab the initial guess and learning rate
        learning_rate = learning_rates[j]
        initial_guess = initial_guesses[i]

        # Run SGD and grab the x, y array of length num_epochs
        x_gd, y_gd = gradient_descent(initial_guess, learning_rate, num_epochs)
        
        # Grab the x,y values for the function under consideration
        x = np.linspace(-10, 8, 500)
        y = 0
        y = [func_y_concave(val) for val in x]

        ax.plot(x, y)
        ax.scatter(x_gd, y_gd, color='red')

        ax.set(xlabel='X', ylabel='Y', 
               title='Cost with initial guess of {}, {} epochs and alpha of {}'
                     .format(initial_guess, num_epochs, learning_rate))

plt.show()

