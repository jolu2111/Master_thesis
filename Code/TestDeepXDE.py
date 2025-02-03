import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print('hello')
# Generate some data
x = np.linspace(0, 10, 100) 
y = np.sin(x)

# Create a plot
plt.plot(x, y)
plt.plot(x, y, 'o')

# Add title and labels
plt.title('Simple Sine Wave Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')


# Show the plot
plt.show()
plt.plot(x, y, '-o')
plt.show()
