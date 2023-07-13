"""
A script to model the spread of COVID-19 in Germany using the SIR model.

author: Fabrizio Musacchio (fabriziomusacchio.vom)
date:

For reproducibility:

conda create -n sir_model_covid19 -y python=3.9
conda activate sir_model_covid19
conda install -y mamba
mamba install -y pandas matplotlib numpy scipy scikit-learn ipykernel notebook ipympl mplcursors
"""
# %% IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
# %% FETCHING AND INSPECTING DATA
# Fetching COVID-19 data for Germany
cases_path ="covid_de.csv"

cases_df = pd.read_csv(cases_path)

# the table contains the cases for each day in each state. We want the total cases for each day in Germany:
cases_df = cases_df.groupby('date').sum().reset_index()

# drop columns that are not needed: state,county,age_group,gender,date:
cases_sub_df = cases_df.drop(columns=['state', 'county', 'age_group', 'gender'])

cases = cases_sub_df["cases"].values.flatten()
deaths = cases_sub_df["deaths"].values.flatten()
recoveries = cases_sub_df["recovered"].values.flatten()
dates = cases_sub_df["date"].values.flatten()
# %% INSPECTING THE DATA
""" plt.plot(dates, cases, label='Confirmed Cases', alpha=0.5)
plt.plot(dates, deaths, label='Deaths', alpha=0.5)
plt.plot(dates, recoveries, label='Recovered', alpha=0.5)
plt.xticks(rotation=45)
plt.legend()
plt.show() """

# define a function, that plot the data for a specific time period:
def plot_data(start_date, end_date):
    start_idx = np.where(dates == start_date)[0][0]
    end_idx = np.where(dates == end_date)[0][0]
    plt.plot(dates[start_idx:end_idx], cases[start_idx:end_idx], label='Confirmed Cases', alpha=1)
    plt.plot(dates[start_idx:end_idx], deaths[start_idx:end_idx], label='Deaths', alpha=0.5)
    plt.plot(dates[start_idx:end_idx], recoveries[start_idx:end_idx], label='Recovered', alpha=0.5, ls="--", c="yellow")
    plt.xticks(rotation=45)
    # determine the number of x-ticks and set the step size accordingly:
    num_ticks = len(dates[start_idx:end_idx])
    step_size = int(num_ticks / 10)
    plt.xticks(np.arange(0, num_ticks, step_size))
    plt.legend()
    plt.show()

plot_data("2020-01-02", "2020-12-02")
# %% PREPROCESSING DATA
# extract from the data above the time-dependent SIR model parameters:
N_population = 83e6
S = N_population - cases
I = cases - deaths - recoveries
R = recoveries


# define a function, that plots S and I for a specific time period:
def plot_SIR(start_date, end_date):
    start_idx = np.where(dates == start_date)[0][0]
    end_idx = np.where(dates == end_date)[0][0]
    plt.plot(dates[start_idx:end_idx], S[start_idx:end_idx], label='Susceptible', alpha=0.5)
    plt.plot(dates[start_idx:end_idx], I[start_idx:end_idx], label='Infected', alpha=0.5)
    plt.xticks(rotation=45)
    # determine the number of x-ticks and set the step size accordingly:
    num_ticks = len(dates[start_idx:end_idx])
    step_size = int(num_ticks / 10)
    plt.xticks(np.arange(0, num_ticks, step_size))
    plt.legend()
    plt.show()

plot_SIR("2022-01-02", "2022-12-02")

# %% PREPROCESSING DATA
# Estimating S and I using the SIR model
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 83000000  # Total population size of Germany (assumed)
I0 = cases[0]  # Initial infected population
R0 = recoveries[0] + deaths[0]  # Initial recovered population

# Solving the SIR model using odeint
def fit_sir_model(beta, gamma):
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.arange(len(cases))

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I

# Fitting the SIR model to the data
def fit_sir_to_data():
    p0 = [0.4, 0.1]  # Initial parameter guess for beta and gamma

    def sumsq(p):
        S, I = fit_sir_model(*p)
        return mean_squared_error(I, cases)

    result = minimize(sumsq, p0, method='Nelder-Mead')
    beta_fit, gamma_fit = result.x

    S_fit, I_fit = fit_sir_model(beta_fit, gamma_fit)

    # Plotting the fitted S and I populations
    """ plt.plot(dates, S_fit, label='Susceptible')
    plt.plot(dates, I_fit, label='Infected')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show() """
    return S_fit, I_fit, beta_fit, gamma_fit

(S_fit, I_fit, beta_fit, gamma_fit) = fit_sir_to_data()

def plot_SIR_model(start_date, end_date):
    start_idx = np.where(dates == start_date)[0][0]
    end_idx = np.where(dates == end_date)[0][0]
    plt.plot(dates[start_idx:end_idx], S_fit[start_idx:end_idx], label='modelled Susceptible', alpha=0.5)
    plt.plot(dates[start_idx:end_idx], I_fit[start_idx:end_idx], label='modelled Infected', alpha=0.5)
    plt.xticks(rotation=45)
    # determine the number of x-ticks and set the step size accordingly:
    num_ticks = len(dates[start_idx:end_idx])
    step_size = int(num_ticks / 10)
    plt.xticks(np.arange(0, num_ticks, step_size))
    plt.legend()
    plt.show()

plot_SIR_model("2020-01-02", "2021-12-02")

# %%