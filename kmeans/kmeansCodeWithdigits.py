from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np 

data = load_digits()
x =data.data
y =data.target

img = x[120].reshape(8,8)
print(y[120])
plt.figure(figsize=(2,2))
plt.imshow(img)
plt.show()