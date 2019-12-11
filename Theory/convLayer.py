import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np
xx, yy = np.meshgrid(np.linspace(0,15,16), np.linspace(0,15,16))

import cv2
img = cv2.imread('mitosisSmall2.png')
img = cv2.resize(img,dsize=(16,16))
img=img[:,:,1]
X =  xx 
Y =  yy
Z =  0*np.ones(X.shape)

data = img
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=1, facecolors=plt.cm.Greys(255-data), shade=False)
ax.grid(True)
for x in np.arange(0,15):
    for y in np.arange(0,15):
        print(x,y)
        ax.text(x+0.5,y+0.5,0.1,'%.1f' % (data[int(x),int(y)]/255), fontweight='bold', fontsize=4, color=[1.0,0.3,1.0], zorder=5, horizontalalignment='center', verticalalignment='center')

#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=plt.cm.Greys(255-data), shade=False)
ax.view_init(elev=90., azim=84)
ax.set_zlim(0,5)
# draw grid
#for y in 
#ax.imshow(img)

xx, yy = np.meshgrid(np.arange(0,5), np.arange(0,5))
filtr = np.array([[255,255, 255,255,255], [0,255, 255,255,255], [0,0,255,255,255], [0,0,0,255,255], [0,0,0,0,255]])
print(xx.shape,yy.shape,filtr.shape)
ax.set_axis_off()
#ax.plot_surface(xx-8, yy+18, 2*np.ones(xx.shape), linewidth=1, zorder=10, facecolors=plt.cm.Greys(filtr), shade=False)
xo_act = -8
yo_act = 12

for x in np.arange(0,5):
    for y in np.arange(0,5):
        print(x,y)
        ax.text(x+xo_act+0.5,y+yo_act+0.5,2.1,'%.1f' % (filtr[int(x),int(y)]/255), fontweight='bold', fontsize=4, color='blue',horizontalalignment='center', verticalalignment='center')
#plt.savefig('conv2d.svg')

# draw grid
for k in range(16):
    plt.plot(xs=[0,15],ys=[k,k], zs=[0,0],color='k', linewidth=0.2)
    plt.plot(xs=[k,k],ys=[0,15], zs=[0,0],color='k', linewidth=0.2)

for k in [0,5]:
    startx=7
    starty=5
    plt.plot(xs=[startx,startx+5],ys=[starty+k,starty+k], zs=[0.01,0.01],color=[1.0,0.3,1.0], linewidth=1.)
    plt.plot(xs=[startx+k,startx+k],ys=[starty+0,starty+5], zs=[0.01,0.01],color=[1.0,0.3,1.0], linewidth=1.)


for k in range(6):
    plt.plot(xs=[xo_act+0,xo_act+5],ys=[yo_act+k,yo_act+k], zs=[0,0],color='k', linewidth=0.2)
    plt.plot(xs=[xo_act+k,xo_act+k],ys=[yo_act+0,yo_act+5], zs=[0,0],color='k', linewidth=0.2)

plt.plot(xs=[startx+5,xo_act+5], ys=[starty, yo_act],color=[1.0,0.3,1.0], linewidth=0.4, linestyle='--')
plt.plot(xs=[startx,xo_act], ys=[starty, yo_act],color=[1.0,0.3,1.0], linewidth=0.4, linestyle='--')
plt.plot(xs=[startx+5,xo_act+5], ys=[starty+5, yo_act+5],color=[1.0,0.3,1.0], linewidth=0.4, linestyle='--')

convresult = np.mean(img[starty:starty+5, startx:startx+5] *  filtr / 255 / 255)
print('Conv result:',convresult)
xo_act2 = -25
yo_act2 = 12
ax.text(startx+xo_act2+0.5,starty+yo_act2+0.5, z=0.0, s='%.1f' % convresult,fontweight='bold', fontsize=4, color='blue',horizontalalignment='center', verticalalignment='center')

#plt.plot(xs=[xo_act2+1+startx,xo_act+5], ys=[starty+yo_act2, yo_act],color=[1.0,0.3,1.0], linewidth=0.4, linestyle='--')
plt.plot(xs=[xo_act2+startx,xo_act], ys=[starty+yo_act2, yo_act],color='b', linewidth=0.4, linestyle='--')
plt.plot(xs=[xo_act2+startx+1,xo_act+5], ys=[starty+yo_act2+1, yo_act+5],color='b', linewidth=0.4, linestyle='--')

for k in [0,5]:
    plt.plot(xs=[xo_act,xo_act+5],ys=[yo_act+k,yo_act+k], zs=[0.01,0.01],color='b', linewidth=1.)
    plt.plot(xs=[xo_act+k,xo_act+k],ys=[yo_act+0,yo_act+5], zs=[0.01,0.01],color='b', linewidth=1.)

xo_act3=xo_act2+startx
yo_act3=yo_act2+starty
for k in [0,1]:
    plt.plot(xs=[xo_act3,xo_act3+1],ys=[yo_act3+k,yo_act3+k], zs=[0.01,0.01],color='b', linewidth=1.)
    plt.plot(xs=[xo_act3+k,xo_act3+k],ys=[yo_act3+0,yo_act3+1], zs=[0.01,0.01],color='b', linewidth=1.)


for k in range(12):
    plt.plot(xs=[xo_act2+0,xo_act2+11],ys=[yo_act2+k,yo_act2+k], zs=[0,0],color='k', linewidth=0.2)
    plt.plot(xs=[xo_act2+k,xo_act2+k],ys=[yo_act2+0,yo_act2+11], zs=[0,0],color='k', linewidth=0.2)
plt.tight_layout()
plt.savefig('conv2d.svg')

