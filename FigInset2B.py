import pandas as pd
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
#import plotly.plotly as py  # tools to communicate with Plotly's server
import csv
import scipy.stats as stats
from matplotlib import pylab

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
    #print sqrt(Var_a),sqrt(Var_b)
    #print "y=ax+b"
    #print "N= %d" % N
    #print "a= %g \pm t_{%d;\alpha/2} %g" % (a, N-2, sqrt(Var_a))
    #print "b= %g \pm t_{%d;\alpha/2} %g" % (b, N-2, sqrt(Var_b))
    #print "R^2= %g" % RR
    #print "s^2= %g" % ss
    return a, b, RR, Var_a, Var_b


#scaling and statistics
center='yes'
compute_res='yes'
histogram_res='yes'
temporal=-11
ptemporal=-11

avx=[]# all cities, all years
avy=[]
xx_tot=[]
yy_tot=[]
label=[]

cnt=0

year=[]
#1969 =2
for yr in range(1969,2016):
    year.append(yr)
#yr=2015
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

    poplog=np.log10(pop)
    wageslog=np.log10(wages)

    xx=poplog
    yy=wageslog

# center data
    if (len(yy)>1 and len(yy)==len(xx) and center=='yes'):
        #print 'lengths=x, y=',len(xx),len(yy)
        av_x=0.
        av_y=0.
        for i in range(len(yy)):
            av_x+=xx[i]
            av_y+=yy[i]
        av_x=av_x/float(len(xx))
        av_y=av_y/float(len(yy))
#xx=xx#-av_x
#        yy=yy-av_y
        avx.append(av_x)
        avy.append(av_y)
#print yr,av_x,av_y

    for i in range(len(yy)):
        xx_tot.append(xx[i])
        yy_tot.append(yy[i])
        label.append(city[i])

    cnt += 2
    f.close()


params = {
    'legend.fontsize': 20,
    'xtick.labelsize':25,
    'ytick.labelsize':25
}
pylab.rcParams.update(params)


plt.plot(avx,avy,'y-',linewidth=5)

gradient, intercept, r_value, var_gr, var_it = linreg(avx,avy)
print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print( "intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print( "R-squared", r_value**2)

tt=avx
tt.sort()
fitx=np.arange(float(tt[0])-0.01,float(tt[-1])+0.01,0.01,dtype=float)
fity=intercept + fitx*gradient
plt.plot(fitx,fity,'k--', linewidth=2, alpha=0.8)

plt.ylabel(r'$\langle \ln Y \rangle$',fontsize=20)
plt.xlabel(r'$\langle \ln N \rangle$',fontsize=20)
plt.tight_layout()
plt.savefig('InsetFig2B.pdf')
plt.show()

