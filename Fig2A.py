import pandas as pd
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import plotly.plotly as py  # tools to communicate with Plotly's server
import csv
import scipy.stats as stats
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb

def linreg(X, Y):
    """
        Summary
        Linear regression of y = ax + b
        Usage
        real, real, real = linreg(list, list)
        Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value
        """
    if len(X) != len(Y):  raise ValueError("unequal length")
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
    meanerror = residual = 0.0
    for x, y in zip(X, Y):
        meanerror = meanerror + (y - Sy/N)**2
        residual = residual + (y - a * x - b)**2
    RR = 1 - residual/meanerror
    ss = residual / (N-2)
    Var_a, Var_b = ss * N / det, ss * Sxx / det
    return a, b, RR, Var_a, Var_b


#scaling and statistics
center='no'
#compute_res='yes'
#histogram_res='yes'
#temporal=-11
#ptemporal=-11

avx=[]# all cities, all years
avy=[]
xx_tot=[]
yy_tot=[]
label=[]

norm = colors.Normalize(vmin=1, vmax=2*47)
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 1


#1969 =2
for yr in range(1969,2016):
    cl='grey'
    mk='o'
    edge_color, color = sm.to_rgba(cnt), sm.to_rgba(cnt+1)
    edge_color='black'
    cnt += 2
    
    count=0
    ii=yr-1967
    f=open('wages.csv', 'r')
    wreader=csv.reader(f,delimiter=',')
    code=[]
    city=[]
    wages=[]
    for row in wreader:
        if (count>5 and count<388):
            #print count,row[0],row[1],row[2]
            code.append(row[0])
            wages.append(float(row[ii]))
            city.append(row[1])
        count+=1
    f.close()

    pop=[]
    for i in range(len(code)):
        pop.append(0.)
    count=0
    g=open('population.csv', 'r')
    preader=csv.reader(g,delimiter=',')
    for row in preader:
        if (count>5 and count<388):
            for i in range(len(code)):
                if (code[i]==row[0]):
                    pop[i]=float(row[ii])
        count+=1
    g.close()

#print yr,len(pop),len(wages)

    poplog=np.log10(pop)
    wageslog=np.log10(wages)

    xx=poplog
    yy=wageslog

#    for i in range(len(poplog)):
#        if (pop[i]>1000. and pop[i]>0. and wages[i]>0.):
#            xx.append(poplog[i])
#            yy.append(wageslog[i])

# center data
    if (len(yy)>1 and len(yy)==len(xx)):
        #print 'lengths=x, y=',len(xx),len(yy)
        av_x=0.
        av_y=0.
        for i in range(len(yy)):
            av_x+=xx[i]
            av_y+=yy[i]
        av_x=av_x/float(len(xx))
        av_y=av_y/float(len(yy))
#xx=xx-av_x
#        yy=yy-av_y
        avx.append(av_x)
        avy.append(av_y)

    for i in range(len(yy)):
        xx_tot.append(xx[i])
        yy_tot.append(yy[i])
        label.append(city[i])


#plt.plot(xx,yy,marker=mk,ms=10,ls='None',markeredgecolor='black',markeredgewidth=1,alpha=0.3)

    cl='grey'
    mk='o'
    
    edge_color, color = sm.to_rgba(cnt), sm.to_rgba(cnt+1)
    edge_color='white'
    cnt += 2
    plt.plot(xx,yy,marker='o',ms=10,ls='None',c=color,markeredgecolor=edge_color,markeredgewidth=.5,alpha=0.6)

    gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
    tt=xx
    tt.sort()
    fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
    fity=intercept + fitx*gradient

    plt.plot(fitx,fity,'k-', linewidth=2, alpha=0.5)

    f.close()

plt.plot(avx,avy,'ys',markeredgecolor='black',ms=10,markeredgewidth=0.5,alpha=0.7)

plt.ylabel('Log Wages',fontsize=20)
plt.xlabel('Log Population',fontsize=20)
plt.tight_layout()
plt.savefig('Fig2A.pdf')
plt.show()
