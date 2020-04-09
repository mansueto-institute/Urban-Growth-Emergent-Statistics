import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import scipy.stats as stats

bbar=0.2
abar=0.1

K1=2.  # This is like Hook's constant or the curvature of the potential that keeps the noise localized 
#K0=5.
K0=(4.*K1)**0.5 # This is the damping
K2=0.  # This is like a mass correction. 
mu=0.
var = 0.0001
sigma= var**0.5

v_error=[]
v_PV=[]


END=100000
#initializations
Dt=.2
p_error=0.
set_point=0.1
PV=set_point
output = 0.
integral=0.
q=0.
p=0.
error=0.
perror=0.
old_v=0.
PPV=0.


v_set=[]
v_q=[]


vec_v=[]
vec_u=[]

time_v=[]
a=[]
b=[]


for i in range(1, END):
    time_v.append(Dt*i)

#### this is the process, or income part of the dynamics,     
    s = np.random.normal(mu, sigma, 1)
    v = 0.05*np.sin(i*Dt/1.) +  s[0]/Dt**0.5
    v=s[0]/Dt**0.5
    dv=v
    #(v-old_v)
    b.append(1.1 +v)
    vec_v.append(v)
    
#### This computes the PID control u
    
    integral = integral + error * Dt
#    derivative = (error - p_error) / Dt
    
    u = K0*error + K1*integral
    #+ K2*derivative

    PV=PV + Dt*u - Dt*dv  # this is b-a, which fluctuates around the set point 
    error = set_point - PV # thus the error is measured relative to the set point.

#    p_error=error  # this just updates the error for the derivative term.

    v_PV.append(PV)
    v_set.append(set_point)
    a.append(1.0 +u) # this is the cost, it has a mean value plus control.

    
#### This is the stochastic system for the error = q 

    q = q + Dt*p
    p = p - Dt*(K1*q+K0*p) + Dt*dv  # this is the stochastic system we should be getting ...
    v_q.append(q)
#    vec_u.append(p+dv)
#######

    v_error.append(error)
    old_v=v

vec=v_q
esigma=np.std(vec)
emu=np.mean(vec)
#expected_var= var/(2.*K1*K0)
### below is expected using dv/dt as a drive and v as Gaussian noise.
expected_var= var/(2.*K1*K0)

print('error var',esigma**2,'expected',expected_var)

fig, ax = plt.subplots()

# the histogram of the data
nbins=30
n, bins, patches = plt.hist(vec, nbins, normed=1, facecolor='grey', alpha=0.75)

# add a 'best fit' Gaussian line
y = mlab.normpdf( bins, emu,esigma)
l = plt.plot(bins, y, 'r-', linewidth=5)
plt.ylabel(r'$P[\epsilon_r]$',fontsize=20)
plt.xlabel(r'$\epsilon_r$',fontsize=20)
plt.xlim(-0.015,0.015)
plt.tight_layout()
plt.savefig('Fig4D.pdf')
plt.show()

