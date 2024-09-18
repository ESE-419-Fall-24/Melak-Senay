import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_experiments = 1000
n_train = 2
x0_values = np.linspace(-1, 1, 100)

# True underlying function
def f(x):
    return x ** 2

# Storage for predictions at x0
g_x0 = np.zeros((num_experiments, len(x0_values)))

for i in range(num_experiments):
    # Generate training data
    x_train = np.random.uniform(-1, 1, n_train)
    y_train = f(x_train)
    
    x1, x2 = x_train
    y1, y2 = y_train
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # predictions
    g_x0[i, :] = slope * x0_values + intercept

#expected value of g(x0)
E_g_x0 = np.mean(g_x0, axis=0)

# true function values at x0_values
f_x0 = f(x0_values)

# Plot
plt.plot(x0_values, E_g_x0, label='E[g(x0)]')
plt.plot(x0_values, f_x0, label='f(x0)')
plt.xlabel('x0')
plt.ylabel('Value')
plt.legend()
plt.title('E[g(x0)] and f(x0) as a function of x0')
plt.show()
