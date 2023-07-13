"""
A script to model the spread of COVID-19 in Germany using the SIR model.

author: Fabrizio Musacchio (fabriziomusacchio.vom)
date:

For reproducibility:

conda create -n sir_model_covid19 -y python=3.9
conda activate sir_model_covid19
conda install -y mamba
mamba install -y pandas matplotlib numpy scipy scikit-learn ipykernel notebook ipympl mplcursors

Acknowledgement:
I acknowledge that the main code is based on this 

* blog post: https://numbersandshapes.net/posts/fitting_sir_to_data_in_python/,  and 
* this documentation page: https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Epidemic_model_SIR.html. 

I have made some modifications to the code to make it more readable and understandable.
"""
# %% IMPORTS
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import odeint
# %% SETTING UP THE MODEL
# set total population: 
N = 1000

# define initial number of infected and recovered individuals, I0 and R0:
I0, R0 = 1, 0

# everyone else is susceptible to infection initially, S0:
S0 = N - I0 - R0

# define contact rate beta and mean recovery rate gamma (in 1/days):
beta, gamma = 0.4, 1./10

# Set a grid of time points (in days):
t = np.linspace(0, 90, 90)  # 90 = 6*15

# create some toy-data. Here: influenza in a British boarding school from 1978. 
data = [1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4]
# %% DEFINING THE MODEL
# the SIR model differential equations:
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
# %% SOLVING THE MODEL
# set initial conditions vector:
y0 = S0, I0, R0

# integrate the SIR equations over the time grid  t:
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T
# %% PLOTTING THE RESULTS
# plot the data on three separate curves for S(t), I(t) and R(t):
scale_Factor =1

fig = plt.figure(2, facecolor='w')
plt.clf()
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/scale_Factor,  alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/scale_Factor,  alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/scale_Factor,  alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(np.arange(0,6*15,6),data,"k*:", label='Original Data')

ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title('SIR Model without fit to the data (initial conditions)')
plt.savefig('SIR Model without fit.png', dpi=200)
plt.show()
# %% FITTING THE MODEL TO THE DATA
# set initial conditions vector:
p=[0.001,1]

# define the sum of squares function:
def sumsq(p, data, t, N):
    beta, gamma = p
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sum_stepSize = int(len(ret[:,1])/len(data))
    return(sum((ret[::sum_stepSize, 1]-data)**2))

# minimize the sum of squares function
msol = minimize(sumsq, p, (data, t, N), method='Nelder-Mead')

beta_fit, gamma_fit = msol.x

# integrate the SIR equations over the time grid t:
ret_fit = odeint(deriv, y0, t, args=(N, beta_fit, gamma_fit))
S_fit, I_fit, R_fit = ret_fit.T
# %% PLOTTING THE RESULTS
# plot the data on three separate curves for S(t), I(t) and R(t):
scale_Factor =1

fig = plt.figure(3, facecolor='w')
plt.clf()
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S_fit/scale_Factor,  alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I_fit/scale_Factor,  alpha=0.5, lw=2, label='Infected (modelled)')
ax.plot(t, R_fit/scale_Factor,  alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(np.arange(0,6*15,6),data,"k*:", label='Original Data')

ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
# ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
ax.text(5, 600,
        (r'$\beta_{model} =$' f'{beta_fit.round(2)}\n' 
         r'$\gamma_{model} =$' f'{gamma_fit.round(2)}\n\n' 
         r'$\Rightarrow \; R_0 =$' f'{(beta_fit/gamma_fit).round(2)}'),
         ha='left', va='bottom', fontsize=10, color='k')

for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title('SIR Model, modelled to the data')
plt.savefig('SIR Model, modelled to the data.png', dpi=200)
plt.show()
# %% END