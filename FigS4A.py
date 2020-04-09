import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
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
    return a, b, RR, Var_a, Var_b


def TwoGaussianFit(bbins, mu, sigma, n):

    dimens=100.
    bestfactor=2.8
    bestmix=0.14
    sssum=100.

    for jj in range(int(dimens)):
        factor =1.2+(jj)/dimens*4.0
        for hh in range(int(dimens)):
            mix=0.03+(hh)/dimens*0.5

            y = mlab.normpdf(bbins, mu, sigma)
            ssigma=sigma/factor
            yy= mlab.normpdf(bbins, mu, ssigma)

            norm=0.
            ndist=[]
            for j in range(len(y)):
                ndist.append(y[j]+mix*yy[j])
                norm+=y[j]+mix*yy[j]
            ndist=ndist/norm/(bbins[1]-bbins[0])


            ssum=0.
            dd=[]
            for j in range(len(bbins)):
                dd.append(n[j]-ndist[j])
                ssum+=abs(n[j]-ndist[j])

            if (ssum< sssum):
                sssum=ssum
                bestfactor=factor
                bestmix=mix
                
    return sssum, bestfactor, bestmix


# 1) Scaling parameters and SAMIs

avx=[]# all cities, all years
avy=[]
xx_tot=[]
yy_tot=[]
label=[]

gradients=[]
pops=[]
intercepts=[]
mean_log_pop=[]
mean_log_wages=[]

year=[]

w, h =47, 382 
Sami = [[0 for x in range(w)] for y in range(h)]
Pops = [[0 for x in range(w)] for y in range(h)]
Wag  = [[0 for x in range(w)] for y in range(h)]

city1=[]

for yr in range(1969,2016):
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
            name.append(row[1])
            code.append(row[0])
            wages.append(float(row[ii])) # all cities year by year
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

    poplog=np.log(pop)
    wageslog=np.log(wages)

    xx=poplog
    yy=wageslog

    xx_av=np.mean(xx)
    yy_av=np.mean(yy)

    # making best fit
    gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
    
    gradients.append(gradient) # beta(t)
    intercepts.append(intercept) # Y0(t)
    mean_log_pop.append(xx_av) # < ln N >(t)
    mean_log_wages.append(yy_av) # < ln Y >(t)

    res=[]
    for i in range(0,len(xx)):
        res.append( yy[i] - (intercept + gradient*xx[i]))

    sigma = np.std(res)
    mu = np.mean(res)
    year.append(yr)
    
    for jj in range(len(xx)):
        Sami[jj][ii-2]=res[jj]
        Pops[jj][ii-2]=pop[jj]
        Wag[jj][ii-2]=wages[jj]


w, h =46, 382 
DeltaLogW= [[0 for x in range(w)] for y in range(h)]
DeltaLogP= [[0 for x in range(w)] for y in range(h)]

w_eta=[]
w_sigma=[]
p_eta=[]
p_sigma=[]


for jj in range(len(xx)):  # This computes temporal growth rates and volatilities from time series for Pop and Wages 
#for jj in range(1):
    S=[]
    T=[]
    for i in range(47):
         S.append(Wag[jj][i])
         T.append(Pops[jj][i])
    vr = np.log(S) # logs of wages
    vt = np.log(T)
    
    #print(S)
    #print(vr)
    
    r=np.diff(vr) #  This is the difference of the logs of WAGES
    rr=np.diff(vt) # This is the difference of the logs of POP

    for ii in range(len(r)):
        DeltaLogW[jj][ii]=r[ii]
        DeltaLogP[jj][ii]=rr[ii]

    # Compute the TEMPORAL averages 
    esigma = np.std(r) # This is the standard deviation of the growth rate of WAGES : sqrt of volatility
    emu = np.mean(r)+0.5*esigma*esigma # This is the (temporal) mean returns for WAGES
    
    tsigma= np.std(rr) # This is the standard deviation of the growth rate of POP : sqrt of volatility
    tmu = np.mean(rr)+0.5*tsigma*tsigma # This is the standard deviation of the growth rate of POP : sqrt of volatility

    w_eta.append(emu) 
    w_sigma.append(esigma)
    #if ( esigma**2>0.005 ):
    #        print(city[jj],esigma**2)
    p_eta.append(tmu)
    p_sigma.append(tsigma)

    if (emu<0.03):
        print(jj,city[jj],'emu=',emu, tmu, 'esigma=',esigma, tsigma)

print ('')
gamma_w=[]
gamma_p=[]
ss=0.
sss=[]
for ii in range(46):  # This computes the ensemble averages (over cities) for each time.
    g = 0.
    p = 0.
    ct=0
    gg=[]
    for j in range (382):
        g+=DeltaLogW[j][ii] # This is the ensemble average of the growth rate of WAGES
        p+=DeltaLogP[j][ii] ## This is the ensemble average of the growth rate of POPULATION
        gg.append(DeltaLogW[j][ii])
        ct+=1
    sigmagg=np.std(gg)
    sss.append(np.std(gg)**2)
    p = p/float(ct)
    g = g/float(ct)

    gamma_w.append(g)
    gamma_p.append(p)

ss_var = np.std(sss)**2
ss=np.mean(sss)


########
    ######## FIGURES #####

edge_color='black'

for jj in range(382):
    yyg=[]
    yyn=[]
    yydelta=[]
    year=[]
    
    for ii in range(len(r)):
        yyg.append(DeltaLogW[jj][ii])
        
        year.append(1970+ii)
    plt.plot(year,yyg,'-',alpha=0.4)
    
plt.plot(year,gamma_w,marker='o',ms=10,ls='None',c='None',markeredgecolor=edge_color,markeredgewidth=1,alpha=0.8)


plt.plot((1969, 2016), (0., 0.), 'k-')

plt.ylabel(r'$\gamma_i(t)$',fontsize=20)
plt.xlabel('Year',fontsize=20)

plt.xlim(1970,2015)
plt.tight_layout()
plt.savefig('FigS4A.pdf')
plt.show()

