# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure(figsize=(10,5))
ax = fig.gca(projection='3d')

quant = 0.05
# Make data.
X = np.arange(-5, 5, quant)
Y = np.arange(-5, 5, quant)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
R2 = np.sqrt((X-2)**2 + (Y-1)**2)
R3 = np.sqrt((X+3)**2 + (Y-2)**2)
Z = 1-np.exp(-R) - 0.5*np.exp(-2*R2)-np.exp(-R3)
L = Z
dX,dY = np.gradient(Z)
print('Z is:',Z.shape)
print( 'dZ is:',dX.shape,dY.shape)

# Plot the surface.

x0 = [2,3]
x1 = [-2,3]
eta = -10
a = list()
b = list()
la = list()
lb = list()

Xnew = list()
Ynew = list()
Lnew = list()

for y in np.arange(-5,5,quant):
    xnew_c = list()
    ynew_c = list()
    lnew_c = list()
    for x in np.arange(-5,5,quant):
        cg = [x,y]
        c = [int(np.round((x+5)/quant)) for x in cg]
        xnew_c.append(x)
        ynew_c.append(y)
        lnew_c.append(L[c[0],c[1]])
    Xnew.append(xnew_c)
    Ynew.append(ynew_c)
    Lnew.append(lnew_c)
Xnew = np.array(Xnew)
Ynew = np.array(Ynew)
Znew = np.array(Lnew)

print('Shape of Xnew: ',np.array(Xnew).shape)        
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                       linewidth=0, antialiased=True)

eta=-20
eta1=-10
eta2=-20
numsteps=50
numsteps_1=20
numsteps_2=50
for k in range(numsteps):
#    a.append(np.copy(x0))
#    b.append(np.copy(x1))
    c_a = [int(np.round((x)/quant)+100) for x in x0]
    c_b = [int(np.round((x)/quant)+100) for x in x1]

    print('Coords in array: ',c_a,c_b)
    print('Loss X0:',L[c_a[0],c_a[1]])
    print('Loss X1:',L[c_b[0],c_b[1]])
    if (k<numsteps_1):
        la.append(np.copy(L[c_a[0],c_a[1]]))
    if (k<numsteps_2):
        lb.append(np.copy(L[c_b[0],c_b[1]]))
    if (k>=20):
        eta2=-10

    if (k<numsteps_1):
        a.append([np.copy(Xnew[c_a[0],c_a[1]]),np.copy(Ynew[c_a[0],c_a[1]])])
    if (k<numsteps_2):
        b.append([np.copy(Xnew[c_b[0],c_b[1]]),np.copy(Ynew[c_b[0],c_b[1]])])
#    lb.append(np.copy(L[c_b[0],c_b[1]]))

    print('X gradient: ',dX[c_a[0],c_a[1]],dY[c_a[0],c_a[1]] )

    if (k<numsteps_1):
        x0[0] = x0[0] + eta1 * dX[c_a[0],c_a[1]]
        x0[1] = x0[1] + eta1 * dY[c_a[0],c_a[1]]

    if (k<numsteps_2):
        x1[0] = x1[0] + eta2 * dX[c_b[0],c_b[1]]
        x1[1] = x1[1] + eta2 * dY[c_b[0],c_b[1]]

a = np.array(a)
b = np.array(b)

print('A:',np.array(a).shape)
print('B:',np.array(b).shape)
ax.plot(a[:,0],a[:,1],la,'r')
ax.plot(b[:,0],b[:,1],lb,'b')
for x in range(numsteps):
    ax.plot(a[x:x+1,0],a[x:x+1,1],la[x:x+1],marker='.', color=(1-x/numsteps,0,0,1))
    ax.plot(b[x:x+1,0],b[x:x+1,1],lb[x:x+1],marker='.', color=(0,0,1-x/numsteps,1))
    if (x<19):
        ax.plot(a[x:x+2,0],a[x:x+2,1],la[x:x+2], color=(1-x/numsteps,0,0,1))
        ax.plot(b[x:x+2,0],b[x:x+2,1],lb[x:x+2], color=(0,0,1-x/numsteps,1))
ax.plot(a[0:1,0],a[0:1,1],la[0:1],'r*')
ax.plot(b[0:1,0],b[0:1,1],lb[0:1],'b*')

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
