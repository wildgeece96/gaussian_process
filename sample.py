import numpy as np 
import matplotlib.pyplot as plt 

from gpr import GPRegression 
from kernel import Kernel

gpr = GPRegression(Kernel, [1.0, 1.0, 1.0, 1.0])  

def truef(X):
    return np.exp(X-2.0) + 1.2

X = np.random.uniform(-3, 3, size=[40, 1])
y = truef(X)  + 0.2*np.random.randn(40,1)

gpr.fit(X,y,num_iter=20, eta=0.05)

X_star = np.linspace(-3, 3, 40).reshape(-1,1)
m, v = gpr.predict(X_star)
m = m.flatten()
v = np.diag(np.abs(v))


plt.figure(figsize=(12,8))
plt.title('The result')
plt.fill_between(X_star.flatten(), m-np.sqrt(v), m +np.sqrt(v))
plt.plot(X_star.flatten(), m , color='red', label='predicted_mean')
plt.scatter(X.flatten(), y.flatten(), label='traindata')
plt.plot(X_star.flatten(), truef(X_star.flatten()), label='true_label', color='purple')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show() 
