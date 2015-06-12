# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <style>
# 
# </style>
# 
# 
# 
# #Gribov model
# ***
# ##Some imports
#  
#    We use <b>L</b>ambda instead <b>l</b>ambda, cause lambda us lambda function in python (lambda x,y: x**2)

# <codecell>

from __future__ import division
from sympy import *
from sympy.interactive import printing
printing.init_printing(use_latex=True)
# commands starting with % are IPython commands
%load_ext sympy.interactive.ipythonprinting

# in older versions of sympy/ipython, use
#%load_ext sympyprinting


#I take Lambda instead lambda, because lambda is lambda function in python
omega, Lambda, D, p, q, k, tau, u1, u2, g1, g2, y, eta, alpha, a, epsilon, xi, d, I_y, I_e = symbols('omega, Lambda, D, p, q, k, tau, u1, u2, g1, g2, y, eta, alpha, a, epsilon, xi, d, I_y, I_e')
DDi = {}

# <markdowncell>

# ##Definition of the graphs
# DDi is a dictionary of graphs.
# ***
# $<\psi^{\dagger},\psi>$-diagrams

# <codecell>

#<psi^dag,psi>
DDi[1] = -(p**2*g1*D*I_y)/(2*d*(1+u1)*y)*(d-1+alpha-(2*alpha)/(1+u1)+(alpha*a*(1-a))/(1+u1)*(4/(1+u1)-d)) - I*omega*g1*alpha*a(1-a)*I_y/(2*(1+u1)**2*y) + tau*D*g1*alpha*a*(1-a)*I_y/(2*(1+u1)**2*y)
DDi[1]

# <codecell>

DDi[2] = -D**2*Lambda**2*I_e*(I*omega/(2*D) - tau - p**2*(d-2)/(2*d))/(2*D*epsilon)
DDi[2]

# <markdowncell>

# ***
# $<\psi^{\dagger},\psi,v>$ -diagrams

# <codecell>

#<psi^dag, psi, v>
DDi[3] = -1*(-1*D**2*Lambda**2)*I*I_e*(2*p+q*(a*d+3-d))/(4*d*D**2*epsilon)
DDi[3]

# <codecell>

DDi[4] = I*g1*alpha*I_y*((p+alpha*q)*(1/d+a-a**2) - 2*a*(1-a)*(2*p+q)/((1+u1)*d))/(2*(1+u1)**2*y)
DDi[4]

# <codecell>

DDi[5] = -u2/D*I*p*g1*D*I_y*(d-1+alpha(1-2*a/(u1+1)))/(2*d*(1+u1)*y)
DDi[5]

# <codecell>

DDi[6] = -u2/D*I*(p+q)*g1*D*I_y*(d-1+alpha-2*alpha*(1-a)/(1+u1))/(2*d*(1+u1)*y)
DDi[6]

# <markdowncell>

# ***
# $<\psi^{\dagger},\psi^{\dagger},\psi>$ -diagrams

# <codecell>

#<psi^dag, psi^dag, psi>
DDi[7] = -D**3*Lambda**3*I_e/(4*D**2*epsilon)
DDi[7]

# <codecell>

DDi[8] = D*Lambda*alpha*g1*I_y*(1-a)**2/(2*y*(1+u1))
DDi[8]

# <codecell>

DDi[9] = -D*Lambda*alpha*g1*a*(1-a)*I_y/(2*(1+u1)**2*y)
DDi[9]

# <markdowncell>

# ***
# $<\psi^{\dagger},\psi,\psi>$ -diagrams

# <codecell>

#<psi, psi, psi^dag>
DDi[10] = D**3*Lambda**3*I_e/(4*D**2*epsilon)
DDi[10]

# <codecell>

DDi[11] = -D*Lambda*alpha*g1*I_y*a**2/(2*y*(1+u1))
DDi[11]

# <codecell>

DDi[12] = -D*Lambda*(-1)*alpha*g1*a*(1-a)*I_y/(2*(1+u1)**2*y)
DDi[12]

# <markdowncell>

# ***
# $<\psi^{\dagger},\psi,v ,v>$ -diagrams

# <codecell>

#<psi, psi^dag, v, v>
DDi[13] = -D**2*Lambda**2*(-1)*I_e/(8*d*D**3*epsilon)
DDi[13]

# <codecell>

DDi[14] = g1*alpha*a*(1-a)*I_y/(2*d*(1+u1)**3*D*y)
DDi[14]

# <codecell>

DDi[15] = -D**2*Lambda**2*I_e/(4*d*D**3*epsilon)
DDi[15]

# <codecell>

DDi[16] = -D*u2*Lambda**2*I_e/(4*D**2*epsilon)
DDi[16]

# <codecell>

DDi[17] = -u2/D*alpha*g1*a*(1-a)*I_y/(2*y*(1+u1)**2)
DDi[17]

# <codecell>

DDi[18] = -u2/D*g1*alpha*a*I_y/(2*d*(1+u1)**2*y)
DDi[18]

# <codecell>

DDi[19] = -u2/D*alpha*g1*I_y*(1-a)/(2*d*(1+u1)**2*y)
DDi[19]

# <codecell>

DDi[20] = u2**2/D**2*g1*D*I_y*(d-1+alpha)/(2*d*(1+u1)*y)
DDi[20]

# <markdowncell>

# ***
# Let substitute $d = 4, I_e=1, I_y=1$  and introduce a new parameter s - dictionary of symmetric coefficients:

# <codecell>

#substituting subs(Lambda**3,Lambda*g2), cause we have Lambda**3
Di = {}
for i in DDi:
    Di[i] = DDi[i].subs(d,4).subs(I_e,1).subs(I_y,1).subs(Lambda**2, g2).subs(Lambda**3,Lambda*g2)

# <codecell>

#symmetric coefficients
s = {}
for i in symbols('s1:21'):
    s[symbols('s1:21').index(i)+1]=i
R = Rational
s[1] = 1
s[2] = R(1/2)
s[3] = 1
s[4] = 1
s[5] = 1
s[6] = 1
s[7] = 2
s[8] = 1
s[9] = 2
s[10] = 2
s[11] = 1
s[12] = 2
s[13] = 2
s[14] = 2
s[15] = 1
s[16] = 1
s[17] = 1
s[18] = 2
s[19] = 2
s[20] = 2
Di[3]

# <markdowncell>

# Now add symmetric coefficients to the graphs:

# <codecell>

#adding symmetric coefficients
for i in Di:
    Di[i] = Di[i]*s[i]

# <codecell>

#Z equations

Z = {}
for i in symbols('z1:9'):
    Z[symbols('z1:9').index(i)+1] = i

#Z of fields, parameters, coupling constants
Z['psidag'], Z['psi'], Z['v'], Z['a'], Z['D'],Z['tau'], Z['Lambda'], Z['u1'], Z['u2'], Z['g1'], Z['g2'] = symbols('Z_psi_dagger, Z_psi, Z_v, Z_a, Z_D, Z_tau, Z_Lambda, Z_u1, Z_u2, Z_g1, Z_g2')

gamma = {}
for i in symbols('gamma1:9'):
    gamma[symbols('gamma1:9').index(i)+1] = i

#gamma of fields, parameters, coupling constants
gamma['psidag'], gamma['psi'], gamma['v'], gamma['a'], gamma['D'], gamma['tau'], gamma['Lambda'], gamma['u1'], gamma['u2'], gamma['g1'], gamma['g2'] = symbols('gamma_psi_dagger, gamma_psi, gamma_v, gamma_a, gamma_D, gamma_tau, gamma_Lambda, gamma_u1, gamma_u2, gamma_g1, gamma_g2')

#equations
e1 = Z['psi']*Z['psidag']-Z[1]
e2 = Z['psi']*Z['psidag']*Z['v']-Z[4]
e3 = Z['psi']*Z['psidag']*Z['a']*Z['v']-Z[5]
e4 = Z['psi']*Z['psidag']*Z['D']-Z[2]
e5 = Z['psi']*Z['psidag']*Z['D']*Z['tau']-Z[3]
e6 = Z['psi']*Z['psidag']**2*Z['D']*Z['Lambda']-Z[6]
e7 = Z['psi']**2*Z['psidag']*Z['D']*Z['Lambda']-Z[7]
e8 = Z['u2']/Z['D']*Z['psi']*Z['psidag']*Z['v']**2-Z[8]

solve((e1,e2,e3,e4,e5,e6,e7,e8), Z['psi'], Z['psidag'], Z['v'], Z['tau'], Z['u2'], Z['Lambda'], Z['D'], Z['a'])[1]

# <codecell>

#<psi^dag, psi>
#From Dyson equation we have minus, so for G_2... <psi psi>1-ir=wZ-pZ-tz - G_2
eq1 = I*omega*gamma[1] - D*p**2*gamma[2] - D*tau*gamma[3]+Di[1]+Di[2]
eq1

# <codecell>

#eq.coeff(var, power) return coefficient of var**power
#expand, cause a*(b+10).coeff(b) == 0
#factor, cause expression is very difficult. But factor doesn't solve this problem.....
#expr[0], cause we have unique solution 
gamma[1] = factor(solve(eq1.expand().coeff(I*omega), gamma[1]))[0]
gamma[2] = factor(solve(eq1.expand().coeff(D*p**2), gamma[2]))[0]
gamma[3] = factor(solve(eq1.expand().coeff(D*tau), gamma[3]))[0]

# <codecell>

#<psi^dag, psi, v>
eq2 = -I*p*gamma[4]-I*a*q*gamma[5]+Di[3]+Di[4]+Di[5]+Di[6]
eq2

# <codecell>

gamma[4] = factor(solve(eq2.expand().coeff(p), gamma[4]))[0]
gamma[5] = factor(solve(eq2.expand().coeff(q), gamma[5]))[0]

# <codecell>

#<psi^dag, psi^dag, psi>
eq3 = D*Lambda*gamma[6]+Di[7]+Di[8]+Di[9]
eq2.expand().coeff(q)

# <codecell>

gamma[6] = factor(solve(eq3, gamma[6]))[0]

# <codecell>

#<psi^dag, psi, psi>
eq4 = -D*Lambda*gamma[7]+Di[10]+Di[11]+Di[12]
eq4

# <codecell>

gamma[7] = factor(solve(eq4, gamma[7]))[0]

# <codecell>

#<psi^dag, psi, v, v>
eq5 = u2/D*gamma[8]+Di[13]+Di[14]+Di[15]+Di[16]+Di[17]+Di[18]+Di[19]+Di[20]
eq5

# <codecell>

gamma[8] = factor(solve(eq5, gamma[8]))[0]
gamma[8]

# <codecell>

for i in gamma:
    Z[i] = 1+collect(factor(simplify(gamma[i].expand())), [g1, g2])

# <codecell>

Z[1]
# i have gamma = sgamma/2

# <codecell>

kapp = 1+g1*alpha*a*(1-a)/((1+u1)**2*y)+g2/(4*epsilon)
kapp

# <codecell>

factor(together(Z[1]*2-kapp))

# <codecell>

Z[2]
# i have gamma = sgamma/2

# <codecell>

Z[3]
#i have gamma = sgamma/2
#and first term with minus !!!

# <codecell>

Z[4]
#i have gamma = sgamma/2
#TODO: g1 term!!!

# <codecell>

Z[5]
#comp TODO

# <codecell>

Z[6]
#comp TODO

# <codecell>

Z[7]
#i have gamma #i have gamma = sgamma/2

# <codecell>

Z[8]

# <codecell>


# <codecell>


