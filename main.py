import numpy as np
import matplotlib.pyplot as plt

nX = 100 # количество x
nT = 100  # количество t

xMax = 1
tMax = 1

eps = 1e-11

y = np.zeros((nX, nT), dtype = float)

x = np.linspace(0, -xMax, nX)
t = np.linspace(0, tMax, nT)

h = float(xMax / (nX-1))
τ = float(tMax / (nT-1))

y[:, 0] = -x
y[0, :] = 0


def p(x):
    return np.arctan(x)


def dp(x):
    return 1 / (1 + x**2)


def f(x, a, b):
    return (x-a)/τ + (p(x) - p(b))/h


def df(x):
    return 1/τ + dp(x)/h


def solve(a, b):
    result = b
    d = eps + 1
    while (d > eps):
        y = result
        result = y - f(y, a, b)/df(y)
        d = abs(y - result)
    return result


for i in range(1, nX):
    for j in range(1, nT):
        y[i, j] = solve(y[i, j-1], y[i-1, j])


fig = plt.figure()
ax = fig.gca(projection = '3d')
t, x = np.meshgrid(t, -x)
surf = ax.plot_surface(-x, t, y, cmap='inferno')
plt.xlabel('x')
plt.ylabel('t')


t_tr = t
x_tr = x

places = np.arange(0, 100, 20)
for place in places:
 plt.plot( t_tr , y[place, : ], label = str(place/100.))
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.title("y(t) при фиксированном x ")
plt.show()
plt.figure()
times = np.arange(1, 101, 20)
for time in times:
 plt.plot( x_tr , y[: , time-1], label = str((time-1)*2./100) + " s")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("y(x) при фиксированном t")
plt.show()