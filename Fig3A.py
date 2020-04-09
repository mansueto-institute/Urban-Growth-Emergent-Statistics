#python code for the plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import lognorm

mu=0.05
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

for sigma in np.arange(0.01,0.6,0.12):
        
    gamma=mu-sigma**2/2.

    time=np.arange(0., 50., dt)
    step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.sqrt(dt)*np.random.normal(0.,1.,(1,n)))
    temp=pd.DataFrame(x0*step.cumprod())
    
    lstep=np.log(step)
    
    stdout=np.std(lstep)/np.sqrt(dt)
    gammaout=np.mean(lstep)/dt
    
    muout=gammaout+stdout**2/2.
    tcrit=sigma**2/(mu-sigma**2/2.)**2

    
    print('gamma',gamma,gammaout)
    print('std',sigma,stdout)
    print('mean',mu,muout)
    print(tcrit)

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
    
#print(gamma,av_gammaout/count)
#print(sigma,av_stdout/count)
#print(mu,av_muout/count)

x.columns=np.arange(0.01,0.6,0.12)

#print(x)
#print(x.loc[0])

#print(s)

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

fig, ax = plt.subplots()

plt.plot(time,x)
#plt.plot(time,y,'-',c='yellow',lw=3)
#plt.plot(time,ym,'-',c='black',lw=3)
#plt.plot(time,yp,'-',c='black',lw=3)

mean=np.exp(gamma*time[-1])
stddev=sigma*np.sqrt(time[-1])

plt.xlabel(r'time',fontsize=20)
plt.ylabel(r'$r(t)$',fontsize=20)

plt.tight_layout()
plt.savefig('Fig3A.pdf')
plt.show()
