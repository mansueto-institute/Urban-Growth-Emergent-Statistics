#python code for the plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import lognorm

mu=0.06
n=500
dt=0.1
x0=1
x=pd.DataFrame()
np.random.seed(9)

count=0.
av_stdout=0.
av_gammaout=0.
av_muout=0.
time=[]

dist=[]

for sigma in np.arange(0.01,0.6,0.00059):
    sigma=0.044721
    
    gamma=mu-sigma**2/2.

    time=np.arange(0., 50., dt)
    step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.sqrt(dt)*np.random.normal(0.,1.,(1,n)))
    temp=pd.DataFrame(x0*step.cumprod())
    s = x0*step.cumprod()
    dist.append(s[-1])

    lstep=np.log(step)
    
    stdout=np.std(lstep)/np.sqrt(dt)
    gammaout=np.mean(lstep)/dt
    
    muout=gammaout+stdout**2/2.

    av_stdout+=stdout
    av_gammaout+=gammaout
    av_muout+=muout

    count+=1.
    x=pd.concat([x,temp],axis=1)


y=[]
ym=[]
yp=[]
for i in range(len(time)):
    y.append(np.exp(gamma*time[i]))
    ym.append(np.exp(gamma*time[i]-1.96*sigma*np.sqrt(time[i])))
    yp.append(np.exp(gamma*time[i]+1.96*sigma*np.sqrt(time[i])))


x.columns=np.arange(0.01,0.6,0.00059)

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

fig, ax = plt.subplots()

num_bins=25
color='blue'
n, bins, patches = plt.hist(dist, num_bins, normed=1, histtype='bar', facecolor=color, linewidth=1, alpha=0.3)

bbins=[]
for j in range(len(n)):
    bbins.append( (bins[j]+bins[j+1])/2.)

mean=np.exp(gamma*time[-1])
stddev=sigma*np.sqrt(time[-1])

distc=lognorm(s=stddev,scale=mean)
xx=np.linspace(0,50,200)
plt.plot(xx,distc.pdf(xx),lw=2,c='red')


plt.xlabel(r'$r$',fontsize=50)
plt.ylabel(r'$P[r]$',fontsize=50)

plt.tight_layout()
plt.savefig('FigInset3B.pdf')
plt.show()
