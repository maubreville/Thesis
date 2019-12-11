import matplotlib.pyplot as plt
import numpy as np

def smoothl1(x):
	y = np.zeros(x.shape)
	y[np.abs(x)<1] = x[np.abs(x)<1]**2 * 0.5
	y[np.abs(x)>=1] = np.abs(x[np.abs(x)>=1]) - 0.5
	return y

def hinge(yHat, y=1):
	return np.maximum(0, 1 - yHat * y)
plt.figure(figsize=(5,3))
x=np.linspace(-4,4,500)
plt.plot(x,np.abs(x), label='L1')
plt.plot(x,0.5*x**2, label='L2')
plt.plot(x,smoothl1(x), label='Smooth L1')
plt.ylim([0, 5])
plt.xlabel('prediction error')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('reg_losses.pdf')

x = np.linspace(0,1,500)
plt.clf()
plt.plot(x,-np.log(x),label='cross-entropy')
plt.plot(x,-(1-x)**2*np.log(x), label='focal loss, $\gamma$=2')
plt.plot(x,-(1-x)**5*np.log(x), label='focal loss, $\gamma$=5')
plt.plot(x, hinge(x), label='hinge')
plt.text(0.7, 1, 'well classified')
plt.ylabel('loss for positive class label')
plt.xlabel('network output')
plt.grid()
plt.legend()
plt.savefig('cla_losses.pdf')
