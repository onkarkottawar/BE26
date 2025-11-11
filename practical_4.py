import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x + 3) ** 2

def grad_f(x):
    return 2 * (x + 3)

def gradient_descent(start_x, learning_rate=0.1, max_iter=50, tol=1e-6):
    x = start_x
    x_history = [x]

    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - learning_rate * grad

        x_history.append(x_new)


        if abs(x_new - x) < tol:
            break

        x = x_new

    return x, f(x), x_history

min_x, min_y, x_steps = gradient_descent(start_x=0)
print("Local minimum occurs at x =", min_x)
print("Minimum value", min_y)

x_values = np.linspace(-6, 2, 100)
y_values = f(x_values)

plt.plot(x_values, y_values, label='f(x) = (x + 3)^2')
plt.scatter(x_steps, [f(x) for x in x_steps], color='red', label='Gradient Descent Steps')
plt.plot(min_x, min_y, 'go', label='Minimum Point')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()
