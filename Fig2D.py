import pandas as pd
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import plotly.plotly as py  # tools to communicate with Plotly's server
import csv
import scipy.stats as stats
import pandas as pd
#from statsmodels.graphics import tsaplots
#from statsmodels.tsa.stattools import acf


def estimated_autocorrelation(x):
    """
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


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
compute_res='no'
histogram_res='yes'
temporal=-11
ptemporal=0

avx=[]# all cities, all years
avy=[]
xx_tot=[]
yy_tot=[]
label=[]

year=[]
w, h =47, 382 
Sami = [[0 for x in range(w)] for y in range(h)] 

#sami=[]
#sami.append([])
city1=[]

#1969=2 column
for yr in range(1969,2016):
#yr=2015
    count=0
    ii=yr-1967
    f=open('wages.csv', 'r')
    wreader=csv.reader(f,delimiter=',')
    code=[]
    city=[]
    wages=[]
    name=[]
    for row in wreader:
        if (count>5 and count<388):
            #print count,row[0],row[1],row[2]
            name.append(row[1])
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
    #print len (xx)


    # making best fit
    gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
    #print yr, "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]"
    #print "intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]"
    #print "R-squared", r_value**2

    res=[]
    for i in range(0,len(xx)):
        res.append( yy[i] - (intercept + gradient*xx[i])) # these are the residuals from scaling
 

    sigma = np.std(res) # standard deviation
    mu = np.mean(res) # mean of residuals over ensemble of cities
    print (yr,'sigma, mu=',sigma, mu)


    year.append(yr)
    for jj in range(len(xx)):
        Sami[jj][ii-2]=res[jj]


for i in range(len(xx)):
    city1=[]
    
    for j in range(47):
        city1.append(Sami[i][j])
    city1=city1-city1[0] # this normalizes all SAMIs to zero at the initial time = 1969.
    plt.plot(year,city1,alpha=0.3)

mean_trajp=[]
mean_trajm=[]
for j in range(47):
    mean_trajp.append(0.02+0.014*np.sqrt(j))
    mean_trajm.append(-0.02-0.014*np.sqrt(j))


# most improved cities, worse cities.

diff=[]
first=[]
last=[]
for i in range(len(xx)):
    diff.append(Sami[i][-1]-Sami[i][0])
    #diff.append(Sami[i][0])
    first.append(Sami[i][0])
    last.append(Sami[i][-1])
xs=[x for y, x in sorted(zip(diff, name))]
ys=[y for y, x in sorted(zip(diff, name))]

#print xs
#print ys



####### Now plot SAMIs over time for selected cities. Best, worse and large.
plt.clf()

for i in range(382):
    s=name[i]

    if s.find("Sunnyvale") != -1:
        #print "Found 'San Jose' in the string."
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='Silicon Valley')
#    plt.annotate('Silicon Valley', xy=(2000, 0.3), xytext=(2010, 0.2),
#            arrowprops=dict(facecolor='blue', shrink=0.05),
#            )

    if s.find("Boulder") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='Boulder CO')

    if s.find("Odessa") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
#        plt.plot(year,city1,alpha=0.8,lw=2,label='Midland TX')

    if s.find("New York") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='New York NY')  

    if s.find("Los Angeles") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='Los Angeles CA')

    if s.find("Chicago") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
#        plt.plot(year,city1,alpha=0.8,lw=2,label='Chicago IL')

    if s.find("Houston") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
#        plt.plot(year,city1,alpha=0.8,lw=2,label='Houston TX')

    if s.find("Anchorage") != -1:
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
#        plt.plot(year,city1,alpha=0.8,lw=2,label='Flint MI')
        
#After World War II, Flint became an automobile manufacturing powerhouse for GM's Buick and Chevrolet divisions, both of which were founded in Flint. However, by the late 1980s the city sank into a deep economic depression after GM closed and demolished several factories in the area, the effects of which remain today.

    if s.find("Las Vegas") != -1:
        #print "Found 'Brownsville' in the string."
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='Las Vegas NV')

    if s.find("Titusville") != -1:
        #print "Found 'San Jose' in the string."
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
#        plt.plot(year,city1,alpha=0.8,lw=2,label='Palm Bay FL')   


    if s.find("Havasu") != -1:
        #print "Found 'San Jose' in the string."
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        #plt.plot(year,city1,alpha=0.8,lw=4,label='Lake Havasu AZ')

    if s.find("McAllen") != -1:
        #print "Found 'San Jose' in the string."
        city1=[]
        for j in range(47):
            city1.append(Sami[i][j])
        plt.plot(year,city1,alpha=0.8,lw=4,label='McAllen TX')

        
#plt.legend()
plt.axhline(linewidth=2, color='k')
plt.xlim((1969,2015))
plt.ylim((-0.5,0.5))
plt.ylabel('SAMIs',fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.tight_layout()
plt.savefig('Fig2D.pdf')
plt.show()

