import numpy as np
import matplotlib.pyplot as plt

def ch1(x):
    return [(1 + x0*x0)*(x - x0) for x0 in np.arange(0, 1.1, .1)]

def ch2(x):
    return [x + t0  for t0 in np.arange(-1, 0.1, .1)]
t_list = np.arange(-1, 1.1, .1)


ch1_list = [ch1(t) for t in t_list]
ch2_list = [ch2(t) for t in t_list]


plt.subplot(1, 2, 1)
plt.ylim(0,1)
plt.xlim(-1,0.1)
plt.plot(ch1_list, t_list)
plt.title('Характеристики при t0 = 0')
plt.ylabel('t')
plt.xlabel('x')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.ylim(0,1)
plt.xlim(0,0.1)
plt.plot(ch2_list, t_list)
plt.title('Характеристики при x0 = 0')
plt.xlabel('x')
plt.grid(True)
plt.show()