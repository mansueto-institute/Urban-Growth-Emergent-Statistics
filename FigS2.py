#python code for the plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import lognorm

mu=0.06
n=500 # time length
nrealiz=1000
nG=3000
dt=0.1
x0=1
x=pd.DataFrame()
np.random.seed(9)

count=0.
av_stdout=0.
av_gammaout=0.
av_muout=0.
time=[]
rg=[]
stdg=[]

dist=[]

#sigma=0.044721
sigma=0.25
gamma=mu-sigma**2/2.



for ii in np.arange(0,nrealiz,1):
    rr=0.
    for jj in np.arange(0,nG,1):
        step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.sqrt(dt)*np.random.normal(0.,1.,(1,n)))
        s = x0*step.cumprod()
        #print (ii,jj, s[-1])
        rr+=s[-1]
    rg.append(rr/nG)

print('length means vector=',len(rg))

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

fig, ax = plt.subplots()

num_bins=20
color='blue'
n, bins, patches = plt.hist(rg, num_bins, normed=1, histtype='bar', facecolor=color, linewidth=1, alpha=0.3)

bbins=[]
for j in range(len(n)):
    bbins.append( (bins[j]+bins[j+1])/2.)

lstep=np.log(rg)

mean= np.mean(lstep)
sigma=np.std(lstep)

print('mean=',mean)
print('sigma=',sigma)

distc=lognorm(s=sigma,scale=np.exp(mean))
xx=np.linspace(min(rg)-1,max(rg)+1,200)
plt.plot(xx,distc.pdf(xx),lw=2,c='red')

plt.xlabel(r'$r_G$',fontsize=50)
plt.ylabel(r'$P[r_G]$',fontsize=50)

plt.tight_layout()
plt.savefig('FigS2.pdf')
plt.show()
