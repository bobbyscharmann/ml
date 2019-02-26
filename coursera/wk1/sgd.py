import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
USE_CONCAVE = False

# Simple parabola
def func_y_concave(x):
    y = x**2 - 4*x + 2
    return y

# Derivative of the func_y defined above
def cost_function_concave(x):
    cost = 2*x - 4
    return cost

# Define another function which does not have concavity. This illustrates the 
# importance of the initial guess in SGD
def func_y_sine(x):
    y = np.sin(x) 
    return y

# Derivative of the above function
def cost_function_sine(x):
    cost = np.cos(x)
    return cost


# Performs stochastic gradient descent using the above defined cost function
# Liberally stolen from:
# https://medium.com/@rrfd/what-is-a-cost-function-gradient-descent-examples-with-python-16273460d634
def gradient_descent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []
    
    x_gd.append(previous_x)
    if USE_CONCAVE:
        y_gd.append(func_y_concave(previous_x))
    else:
        y_gd.append(func_y_sine(previous_x))

    for i in range(epoch):
        current_x = 0
        current_y = 0
        if USE_CONCAVE:
            current_x = previous_x - learning_rate * (cost_function_concave(previous_x))
            current_y = func_y_concave(current_x)
        else:
            current_x = previous_x - learning_rate * (cost_function_sine(previous_x))
            current_y = func_y_sine(current_x)

        x_gd.append(current_x)
        y_gd.append(current_y)

        previous_x = current_x

    return x_gd, y_gd


initial_guesses = [-1, 6]
learning_rates = [0.1, 0.2, 0.4]
num_epochs = 20
gs = gridspec.GridSpec(len(initial_guesses), len(learning_rates))
fig = plt.figure(tight_layout=True)

for i in range(len(initial_guesses)):
    for j in range(len(learning_rates)):
        ax = fig.add_subplot(gs[i, j])
        learning_rate = learning_rates[j]
        initial_guess = initial_guesses[i]
        x_gd, y_gd = gradient_descent(initial_guess, learning_rate, num_epochs)
            
        x = np.linspace(-10, 8, 50)
        y = 0
        if USE_CONCAVE:
            y = [func_y_concave(val) for val in x]
        else:
            y = [func_y_sine(val) for val in x]

        ax.plot(x, y)
        ax.scatter(x_gd, y_gd, color='red')

        ax.set(xlabel='X', ylabel='Y', 
               title='Cost with initial guess of {}, {} epochs and alpha of {}'
                     .format(initial_guess, num_epochs, learning_rate))

plt.show()

