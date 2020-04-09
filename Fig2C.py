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
    mean_log_pop.append(xx_av) # < ln N >(t) center N
    mean_log_wages.append(yy_av) # < ln Y >(t) center Y

    res=[]
    for i in range(0,len(xx)):
        res.append( yy[i] - (intercept + gradient*xx[i]))

    sigma = np.std(res) # ensemble average of residuals
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


######## Figures


#### Figure BM2 #####
#fig, ax = plt.subplots()

xx=[]
year=[]
aux2=0.

for ii in range(len(r)): # over time
    
    year.append(1969+ii)
    aux=0.
    count=0
    
    for jj in range(382): # for each city
        aux+=(Sami[jj][ii]-Sami[jj][0] )**2
        #aux+=(DeltaLogW[jj][ii]-gamma_w[ii]) - gradients[ii]*(DeltaLogP[jj][ii] -gamma_p[ii])
        aa=0.
        for iii in range(ii):
            aa+=(DeltaLogW[jj][iii]-gamma_w[iii]) - gradients[iii]*(DeltaLogP[jj][iii] -gamma_p[iii])
        #((DeltaLogW[jj][ii]-gamma_w[ii]) - gradients[ii]*(DeltaLogP[jj][ii] -gamma_p[ii]))**2
        #print(ii,aa)
        #aux+=aa**2
        count+=1
    #print(aux)
    aux2=aux/float(count)
        
    xx.append(aux2)    


#ax.axvspan(1969.91667, 1970.8333, alpha=0.3, color='grey')
#ax.axvspan(1973.8333, 1975.167, alpha=0.3, color='grey')
#ax.axvspan(1969+11.0, 1969+11.5, alpha=0.3, color='grey')
#ax.axvspan(1969+12.5, 1969+13.8333, alpha=0.3, color='grey')
#ax.axvspan(1969+21.5, 1969+22.167, alpha=0.3, color='grey')
#ax.axvspan(1969+32.167, 1969+32.8333, alpha=0.3, color='grey')
#ax.axvspan(1969+38.8333, 1969+40.416667, alpha=0.3, color='grey')

#xxlog=np.log(xx)

# global best fit
gradient, intercept, r_value, var_gr, var_it = linreg(year,xx)
#print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
#print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
#print("R-squared", r_value**2)

tt=year
tt.sort()
fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
fity=intercept + fitx*gradient
#plt.plot(fitx,fity,'-', c='black', linewidth=2, alpha=0.8,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')



# local best fits
year1=[]
xx1=[]
for ii in range(18):
    year1.append(year[ii])
    xx1.append(xx[ii])

gradient, intercept, r_value, var_gr, var_it = linreg(year1,xx1)
#print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
#print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
#print("R-squared", r_value**2)


tt=year1
tt.sort()
fitx=np.arange(float(tt[0])-5,float(tt[-1])+5,0.1,dtype=float)
fity=intercept + fitx*gradient
#plt.plot(fitx,fity,'-', c='red', linewidth=2, alpha=0.8,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')


year2=[]
xx2=[]
for ii in range(22,len(r)):
    year2.append(year[ii])
    xx2.append(xx[ii])

gradient, intercept, r_value, var_gr, var_it = linreg(year2,xx2)
#print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
#print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
#print("R-squared", r_value**2)


tt=year2
tt.sort()
fitx=np.arange(float(tt[0])-9,float(tt[-1])+5,0.1,dtype=float)
fity=intercept + fitx*gradient
#plt.plot(fitx,fity,'-', c='red', linewidth=2, alpha=0.8,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')


#plt.plot(year,xx,'bo',ms=8, alpha=0.4)

#plt.ylabel(r'$\langle \Delta_i^2 \rangle$',fontsize=20)
#plt.xlabel('Year',fontsize=20)
#plt.tight_layout()
#plt.xlim(1968,2016)
#plt.ylim(0.,0.0)
#plt.savefig('Mean_Square_Displacement_Wages_Total.png', format='png', dpi=1200)
#plt.show()








#### Figure BM2 #####
#plt.clf()
#fig, ax = plt.subplots()

for jj in range(382): # for each city 
    year=[]
    yydelta=[]
    aux2=0.
    
    for ii in range(len(r)): # over time
        aux2= Sami[jj][ii]-Sami[jj][0]
        
        yydelta.append(aux2) # this is DELTA SAMIs IN TIME
        
        year.append(1970+ii)
#    plt.plot(year,yydelta,'-',alpha=0.4)

mean_trajp=[]
mean_trajm=[]

for j in range(len(r)):
    mean_trajp.append(np.sqrt(0.00120544646369 +0.00108420769819*j))
    mean_trajm.append(-np.sqrt(0.00120544646369 +0.00108420769819*j))
    
#plt.plot(year,mean_trajp,'r-',linewidth=3)
#plt.plot(year,mean_trajm,'r-',linewidth=3)


#plt.plot((1969, 2016), (0., 0.), 'k-')

#plt.xlabel('Year',fontsize=20)
#plt.ylabel(r'$t \Delta_i(t)$',fontsize=20)

#plt.xlim(1970,2015)
#plt.ylim(-1.,1.)
#plt.tight_layout()
#plt.savefig('Fig5B.pdf')
#plt.show()






### Figure variance prediction ###

#plt.clf()
#fig, ax = plt.subplots()

sigmas=[]
av_sigma=0.
for jj in range(382):
    sigmas.append(w_sigma[jj]**2)
    av_sigma+=w_sigma[jj]**2

    
av_sigma=av_sigma/float(len(w_sigma))
sigsig=0.
for jj in range(382):
    sigsig+=((av_sigma-sigmas[jj])**2)
sigsig=sigsig/float(len(w_sigma)-1)
sigsig=np.sqrt(sigsig)
print(av_sigma, sigsig)
#ax.axhspan(av_sigma-sigsig,av_sigma+sigsig, alpha=0.2, color='grey')
#ax.axhspan(av_sigma,av_sigma, alpha=1.0, color='blue')

print('standard deviation',ss,np.sqrt(ss_var))
#ax.axhspan(ss-np.sqrt(ss_var),ss+np.sqrt(ss_var), alpha=0.2, color='grey') # not right.
#ax.axhspan(ss,ss, alpha=1.0, color='green')

#ax.axhspan(gradient-np.sqrt(var_gr),gradient-np.sqrt(var_gr), alpha=0.2, color='grey')
#ax.axhspan(gradient,gradient, alpha=1.0, color='red')

#plt.plot(w_eta,sigmas,'bo',alpha=0.2)
#plt.ylabel(r'$\sigma^2_i$',fontsize=20)
#plt.xlabel(r'${\bar \eta}_i$',fontsize=20)
#plt.tight_layout()
#plt.savefig('Variance_Prediction.png', format='png', dpi=1200)
#plt.show()

####### Now histograms of SAMIs to show increase in varience over time ####

#1) Average of SAMIs in time, histogram and best fit as Gaussiana and as sum of Gaussians

# build vector of all SAMIs

#plt.clf()
fig, ax = plt.subplots()

all_res=[]
for i in range(len(year)):
    for jj in range (len(pop)): 
        all_res.append(Sami[jj][i])

#all_res=all_res
#/np.sqrt(len(year))

sigma = np.std(all_res)
mu = np.mean(all_res)
num_bins=60

n, bins, patches = plt.hist(all_res, num_bins, normed=1, facecolor='grey', alpha=0.5)  #############
    
bbins=[]
for j in range(len(n)):
    bbins.append( (bins[j]+bins[j+1])/2.)

sssum, bestfactor, bestmix = TwoGaussianFit(bbins, mu, sigma, n)  #### this computes the 2 Gaussian fit
                
print( 'mean= ',mu, 'sigma= ',sigma, 'second amplitude= ',bestmix, 'sec sigma= ',sigma/bestfactor )

y = mlab.normpdf(bbins, mu, sigma)
ssigma=sigma/bestfactor
yy= mlab.normpdf(bbins, mu, ssigma)
        
norm=0.
ndist=[]
for j in range(len(y)):
    ndist.append(y[j]+bestmix*yy[j])
    norm+=y[j]+bestmix*yy[j]
ndist=ndist/norm/(bbins[1]-bbins[0])
plt.plot(bbins, ndist, 'r--',lw=2)

dd=[]
for j in range(len(bbins)):
    dd.append(n[j]-ndist[j])


plt.plot(bbins, y, 'b-',lw=3)
plt.xlim(-1,1)
plt.ylim(0.,3.5)
plt.xlabel(r'$\xi_i(t)$',fontsize=20)
plt.ylabel('Probability',fontsize=20)
plt.tight_layout()
plt.savefig('Fig2C.pdf')
plt.show()
