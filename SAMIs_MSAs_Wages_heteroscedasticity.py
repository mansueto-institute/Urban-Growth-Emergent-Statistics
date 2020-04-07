import pandas as pd
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import plotly.plotly as py  # tools to communicate with Plotly's server
import csv
import scipy.stats as stats
import statsmodels.api as sm



def quartiles(dataPoints):
    # check the input is not empty
    if not dataPoints:
     raise StatsError('no data points passed')
        # 1. order the data set
     sortedPoints = sorted(dataPoints)
        # 2. divide the data set in two halves
     mid = len(sortedPoints) / 2

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


#1969 =2
for yr in range(2015,2016):
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
    if (len(yy)>1 and len(yy)==len(xx) and center=='yes'):
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
#print yr,av_x,av_y

    for i in range(len(yy)):
        xx_tot.append(xx[i])
        yy_tot.append(yy[i])
        label.append(city[i])

# plot data
    cl='grey'
    mk='o'

#    plt.plot(xx,yy,marker=mk,ms=10,ls='None',markeredgecolor='white',markeredgewidth=1,alpha=0.5)
    f.close()


# making best fit
gradient, intercept, r_value, var_gr, var_it = linreg(xx_tot,yy_tot)
print ("Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print ("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print( "R-squared", r_value**2 )


# show models and best fit
tt=xx_tot
tt.sort()
fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
fity=intercept + fitx*gradient
#fityy=intercept + fitx
fityyy= intercept+ 7./6.*fitx

#plt.plot(fitx,fity,'r-', linewidth=2, alpha=0.8,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')
    #plt.plot(fitx,fityy,'k-', linewidth=2, alpha=0.5,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')
    #plt.plot(fitx,fityyy,'y-', linewidth=6, alpha=0.5,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')


#plt.ylabel('Log Wages',fontsize=20)
#plt.xlabel('Log Population',fontsize=20)
#plt.show()

#compute residuals (SAMIs)
res=[]
for i in range(0,len(xx)):
        res.append( yy[i] - (intercept + gradient*xx[i]))

#plt.plot(xx,res,marker=mk,ms=10,ls='None',markeredgecolor='black',markeredgewidth=1,alpha=0.5)

print('mean residuals:',np.mean(res))

rgradient, rintercept, rr_value, var_gr, var_it = linreg(xx_tot,res)
print ("Gradient=", rgradient, ", 95 % CI = [",rgradient- 2.*np.sqrt(var_gr),",",rgradient+2.*np.sqrt(var_gr),"]")
print ("intercept=", rintercept, ", 95 % CI = [",rintercept- 2.*np.sqrt(var_it),",",rintercept+2.*np.sqrt(var_it),"]")
print( "R-squared", rr_value**2 )



xs=[x for y, x in sorted(zip(res, xx))]
ys= [y for y, x in sorted(zip(res, xx))]

plt.plot(xs,ys,marker=mk,ms=10,ls='None',markeredgecolor='white',markeredgewidth=1,alpha=0.5)
plt.plot((min(xx)-0.2,max(xx)+0.2),(0.0,0.),'k-')



md=np.median(xs)
qq=[]
for i in range(len(xs)):
    if (xs[i]<= md):
        qq.append(xs[i])

mqq=np.median(qq)

qq=[]
for i in range(len(xs)):
    if (xs[i]> md):
        qq.append(xs[i])

Mqq=np.median(qq)
#print('qq',mqq,md,Mqq)

sigma_m=0.
sigma_mdm=0.
sigma_mdM=0.
sigma_M=0.
x_m=0.
x_mdm=0.
x_mdM=0.
x_M=0.


n_m=0.
n_mdm=0.
n_mdM=0.
n_M=0.

for i in range(len(xs)):
    if (xs[i]<=mqq):
        sigma_m+=ys[i]**2
        x_m+=xs[i]
        n_m+=1.
    if (xs[i]>mqq and xs[i]<=md):
        sigma_mdm+=ys[i]**2
        x_mdm+=xs[i]
        n_mdm+=1.

    if (xs[i]>md and xs[i]<=Mqq):
        sigma_mdM+=ys[i]**2
        x_mdM+=xs[i]
        n_mdM+=1.

    if (xs[i]>Mqq):
        sigma_M+=ys[i]**2
        x_M+=xs[i]
        n_M+=1.

sigmas=[]
xs=[]

sigma_m=np.sqrt(sigma_m/n_m)
sigmas.append(sigma_m)
sigma_mdm=np.sqrt(sigma_mdm/n_mdm)
sigmas.append(sigma_mdm)
sigma_mdM=np.sqrt(sigma_mdM/n_mdM)
sigmas.append(sigma_mdM)
sigma_M=np.sqrt(sigma_M/n_M)
sigmas.append(sigma_M)

x_m=x_m/n_m
xs.append(x_m)
x_mdm=x_mdm/n_mdm
xs.append(x_mdm)
x_mdM=x_mdM/n_mdM
xs.append(x_mdM)
x_M=x_M/n_M
xs.append(x_M)

gradient, intercept, r_value, var_gr, var_it = linreg(xs,sigmas)
print ("Gradient sigmas=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print ("intercept sigmas=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print( "R-squared sigmas", r_value**2 )

#model = sm.OLS(sigmas, xs).fit()
# Print out the statistics
#model.summary()




print('sigmas',sigma_m,sigma_mdm,sigma_mdM,sigma_M)
print('x',x_m,x_mdm,x_mdM,x_M)

plt.plot((min(xx)-0.2,max(xx)+0.2),(0.105,0.105),'k--')
plt.plot((x_m,x_mdm,x_mdM,x_M),(sigma_m,sigma_mdm,sigma_mdM,sigma_M),'ro',ms=10)
plt.plot((mqq,mqq),(-0.4,0.4),'r--')
plt.plot((md,md),(-0.4,0.4),'r--')
plt.plot((Mqq,Mqq),(-0.4,0.4),'r--')



plt.ylabel('residuals',fontsize=20)
plt.xlabel('Log Population',fontsize=20)
plt.show()
