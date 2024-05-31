import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('stack.txt', unpack=True)
dx = data[0]
dy = data[1]
g1 = data[2]
g2 = data[3]



def get_xy(g1, g2):
    x = 1
    if g1 > 0:
        y = np.tan(np.arctan(g2/g1) / 2)
    else:
        y = - 1 / np.tan(np.arctan(g2/g1) / 2)
    
    scale = np.sqrt(g1**2 + g2**2) / np.sqrt(x**2 + y**2)
    x = x * scale
    y = y * scale
    
    return [x, y]



plt.figure(figsize=(7, 7))

for k in range(len(dx)):
    xy = get_xy(g1[k], g2[k])
    plotscale = 2e-7
    x = xy[0] * plotscale
    y = xy[1] * plotscale
    plotx = [dx[k] - x, dx[k] + x]
    ploty = [dy[k] - y, dy[k] + y]
    plt.plot(plotx, ploty, c='tomato', linewidth=2)


plt.scatter(0, 0, s=80)
plt.xlabel('dx [Mpc]')
plt.ylabel('dy [Mpc]')
plt.savefig('stack.png')
plt.clf()
