import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-5,5,100)

def relu(x):
    x=np.copy(x)
    x[x<0] = 0
    return x

def sign(x):
    y=np.copy(x)
    y[x<0]=-1
    y[x>=0]=1
    return y

def step(x):
    y=np.copy(x)
    y[x<0]=0
    #y[x==0]=0.5
    y[x>=0]=1
    return y

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def leakyrelu(x,eps=0.05):
    x=np.copy(x)
    x[x<0] *= eps
    return x


fig1 = plt.figure(figsize=(5,5))
ax1 = plt.gca()
plots=[]

fig2 = plt.figure(figsize=(5,5))
ax2 = plt.gca()

linestyles = ['-','--','-.',':','-','--']
for fcn,title,ls in zip([step, sign, sigmoid, tanh, relu, leakyrelu],['step(x)', 'sign(x)', '$\sigma$(x)','tanh(x)','ReLu(x)', '$\\mathrm{LReLu}_{\\alpha=0.05}$(x)',], linestyles):
    df = np.diff(fcn(x))
    dx = np.diff(x)
    plots+=[ax1.plot(x, fcn(x), label=title, linestyle=ls)]
    ax1.set_xlabel('input value x')
    ax1.set_ylabel('f(x)')
    ax1.set_ylim([-1,5])
    ax1.grid(True)

    plots+=[ax2.plot(x[:-1], df/dx, label=title, linestyle=ls)]
    ax2.set_xlabel('input value x')
    ax2.set_ylabel('df/dx')
    ax2.set_ylim([-0.4,2])
    ax2.grid(True)


fig1.legend(bbox_to_anchor=(0.15, 0.96), loc='upper left', framealpha=1.0)
fig2.legend(bbox_to_anchor=(0.15, 0.96),loc='upper left',framealpha=1.0)
print(plots)
plots[-1][0].set_linestyle('--')
plots[-2][0].set_linestyle('--')
fig1.tight_layout()
fig2.tight_layout()

fig1.savefig('activationfunctions.pdf')
fig2.savefig('activationfunctions_derivative.pdf')
