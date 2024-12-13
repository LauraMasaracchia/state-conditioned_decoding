"""
Script to run the neuron-state interaction analysis.
The analysis should be run per rat.
It outputs the results of the analysis in a '.pkl' file, and plots the results.

This script performs:
1. interaction analysis with statistical testing
2. estimation of the coefficients' confidence interval via bootstrapping

Author: Laura Masaracchia
Email: laurama@cfin.au.dk
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from .utils.analysis import convert_vp_to_gamma


# -------------------------------------------------------------------
# ------------------------ hyperparameters --------------------------
# choose rat name. Select from 'Buchanan', 'Stella', 'Mitt', 'Barat'
rat_name = 'Buchanan'
print(rat_name)

# Analysis hyperparameters
n_perms = 100000


# get HMM info
if rat_name == 'Mitt':
    dirdiag = 10000000
else:
    dirdiag = 1000000000

lags = 7
n_states = 2

# ----------------------------------------------------------------------------
# ----------------------------  paths to data --------------------------------

data_folder_name = './data/preprocessed_data/'
filename_behav_info = 'behav_info.pkl'
neural_density_around_resp_filename = '%s_spike_density_G10ms_250Hz_response.pkl' % rat_name

results_directory = './results/'
states_around_resp_filename = 'states_around_response_%s_dd%d_l%d_K%d.pkl' % (rat_name, dirdiag, lags, n_states)

# ----------------------------------------------------------------------------
# -------------------------------- load data ---------------------------------

# load behavioural info
with open(os.path.join(data_folder_name, rat_name, filename_behav_info), 'rb') as fp:
    behav_info = pickle.load(fp)

# load states
with open(os.path.join(results_directory, rat_name, states_around_resp_filename), 'rb') as fp:
    states_response_dict = pickle.load(fp)

# load neural activity
with open(os.path.join(data_folder_name, rat_name, neural_density_around_resp_filename), 'rb') as fp:
    units_density_response = pickle.load(fp)


poke_out_time = behav_info['poke_out_time']
# success label is with +1 if success, -1 if fail
success_label = behav_info['success_label']

states_around_response = states_response_dict['VP']
gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
# get states into +1 and -1
gamma = gamma_vp_response[:, 0, :] * 2.0 - 1.0

n_trials, n_units, n_time_points = units_density_response.shape

# ----------------------------------------------------------------------------------------------
# ---------------------- run regression per neuron and per time point --------------------------
y = success_label
X_units = units_density_response

# standardise units per time point and per trial, across neurons
X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

# initialization
n_features = 3
betas = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
mu = np.empty(shape=(n_time_points, n_units, n_perms + 1))
mean_error = np.empty(shape=(n_time_points, n_units, n_perms + 1))

units_mult_gamma1 = X_units * gamma[:, np.newaxis, :]

# the first one is the basic statistic: run the regression per neuron, no swapping
# concatenate along the feature axis units activity, states activation and their multiplication
# base statistics
for j in range(n_units):
    X = np.concatenate((X_units[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], gamma[:,np.newaxis,:]), axis=1)
    # first run the unpermuted regression
    for t in range(n_time_points):
        model = Ridge(alpha=0.001)
        model.fit(X[:, :, t], y)
        betas[t, (j * n_features):(j * n_features) + n_features, 0] = model.coef_
        mu[t, j, 0] = model.intercept_
        mean_error[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model.coef_) - model.intercept_))


# then we do with the permutations swapping gammas, the same index for all neurons and for all time points.
# swap for each iteration
for k in range(1, n_perms + 1):
    print(k)
    # for permutations
    swapped_gamma = gamma.copy()
    swap_idx = np.random.choice(np.arange(n_trials), int(n_trials/2),replace=False)
    swapped_gamma[swap_idx, :] = - gamma[swap_idx, :]

    units_mult_gamma1 = X_units * swapped_gamma[:, np.newaxis, :]

    for j in range(n_units):
        # resample indices with replacement
        X = np.concatenate((X_units[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], swapped_gamma[:,np.newaxis,:]), axis=1)
        for t in range(n_time_points):
            model = Ridge(alpha=0.001)
            model.fit(X[:, :, t], y)
            betas[t, (j * n_features):(j * n_features) + n_features, k] = model.coef_
            mu[t, j, k] = model.intercept_
            mean_error[t, j, k] = np.mean(np.abs(y - np.matmul(X[:, :, t], model.coef_) - model.intercept_))


# ----------------------------------------------------------------------------------------------
# ---------------------------------- save results ----------------------------------------------

multiple_regr_dict = {'betas_ridge': betas, 'mu_ridge': mu, 'err_ridge': mean_error}

output_filename = "%s_vp_interaction_analysis_perm_random_states_perm_%d.pkl"%(rat_name,n_perms)
with open(os.path.join(results_directory, rat_name, output_filename), 'wb') as fp:
    pickle.dump(multiple_regr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

print('%s - INTERACTION analysis finished and results stored - %s' % (rat_name, output_filename))

# ----------------------------------------------------------------------------------------------
# ---------------------------------- plot results ----------------------------------------------

# plot average prediction error across neurons
base_err = multiple_regr_dict['err_ridge'][:, :, 0]
random_err = multiple_regr_dict['err_ridge'][:, :, 1:]
# plot errors and accuracy
fig, ax = plt.subplots()
ax.plot(np.arange(n_time_points), np.mean(base_err, 1), color='navy', label='HMM MAE ridge')
ax.plot(np.arange(n_time_points), np.mean(np.mean(random_err, 2), 1), color='steelblue',label='rand MAE ridge')
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
plt.legend()
ax.set_ylim([0.2, 0.7])
ax2 = ax.twinx()
err_diff = np.mean(np.mean(random_err, 2), 1) - np.mean(base_err, 1)
ax2.bar(np.arange(n_time_points), err_diff, width=0.2, color='gray', alpha=0.5)
plt.xlabel('time')
plt.ylabel('prediction')
plt.title('average prediction accuracy/error %s' % rat_name)
plt.show()


# compute pvalue
pvalue_ridge = np.sum(base_err[:, :, np.newaxis] >= random_err, axis=2) / n_perms


# plot pvalue
fig, ax = plt.subplots()  # (nrows=3)
im2 = ax.imshow(pvalue_ridge.transpose(1, 0), cmap='plasma', vmin=0, vmax=0.05)
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
fig.colorbar(im2, ax=ax, location='bottom', orientation='horizontal')
fig.suptitle('pvalues, %s' % rat_name)
plt.show()

# ----------------------------------------------------------------------------------------------
# --------------------- bootstrapping for confidence interval ----------------------------------
# ----------------------------------------------------------------------------------------------

# choose rat name. Select from 'Buchanan', 'Stella', 'Mitt', 'Barat'
rat_name = 'Buchanan'
print(rat_name)

# Analysis hyperparameters
n_bootstrap = 100

# get HMM info
if rat_name == 'Mitt':
    dirdiag = 10000000
else:
    dirdiag = 1000000000

lags = 7
n_states = 2

# ----------------------------------------------------------------------------
# ----------------------------  paths to data --------------------------------

data_folder_name = './data/preprocessed_data/'
filename_behav_info = 'behav_info.pkl'
neural_density_around_resp_filename = ''

results_directory = './results/'
states_around_resp_filename = 'states_around_response_%s_dd%d_l%d_K%d.pkl' % (rat_name, dirdiag, lags, n_states)

# ----------------------------------------------------------------------------
# -------------------------------- load data ---------------------------------

# load behavioural info
with open(os.path.join(data_folder_name, rat_name, filename_behav_info), 'rb') as fp:
    behav_info = pickle.load(fp)

# load states
with open(os.path.join(results_directory, rat_name, states_around_resp_filename), 'rb') as fp:
    states_response_dict = pickle.load(fp)

# load neural activity
with open(os.path.join(data_folder_name, rat_name, neural_density_around_resp_filename), 'rb') as fp:
    units_density_response = pickle.load(fp)


poke_out_time = behav_info['poke_out_time']
# success label is with +1 if success, -1 if fail
success_label = behav_info['success_label']

states_around_response = states_response_dict['VP']
gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
# get states into +1 and -1
gamma = gamma_vp_response[:, 0, :] * 2.0 - 1.0

n_trials, n_units, n_time_points = units_density_response.shape

# ----------------------------------------------------------------------------------------------
# ---------------------- run regression per neuron and per time point --------------------------
y = success_label
X_units = units_density_response

# standardise units per time point and per trial, across neurons
X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

# initialization
n_features = 3
betas = np.empty(shape=(n_time_points, n_units * n_features, n_bootstrap + 1))
mu = np.empty(shape=(n_time_points, n_units, n_bootstrap + 1))
mean_error = np.empty(shape=(n_time_points, n_units, n_bootstrap + 1))

units_mult_gamma1 = X_units * gamma[:, np.newaxis, :]

# the first one is the basic statistic: run the regression per neuron, no swapping
# concatenate along the feature axis units activity, states activation and their multiplication
# base statistics
for j in range(n_units):
    X = np.concatenate((X_units[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], gamma[:,np.newaxis,:]), axis=1)
    # first run the unpermuted regression
    for t in range(n_time_points):
        model = Ridge(alpha=0.001)
        model.fit(X[:, :, t], y)
        betas[t, (j * n_features):(j * n_features) + n_features, 0] = model.coef_
        mu[t, j, 0] = model.intercept_
        mean_error[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model.coef_) - model.intercept_))


# for the bootstrap, resample with replacement at each iteration
for k in range(1, n_bootstrap + 1):
    print(k)
    # for bootstrap
    idx = np.random.choice(np.arange(n_trials), n_trials, replace=True)
    gamma_boot = gamma[idx]
    X_units_boot = X_units[idx]
    y_boot = y[idx]

    units_mult_gamma_boot = X_units_boot * gamma_boot[:, np.newaxis, :]

    for j in range(n_units):
        # resample indices with replacement
        X_boot = np.concatenate((X_units_boot[:, j, :][:, np.newaxis, :], units_mult_gamma_boot[:, j, :][:, np.newaxis, :],
                                gamma_boot[:, np.newaxis, :]), axis=1)
        for t in range(n_time_points):
            model = Ridge(alpha=0.001)
            model.fit(X_boot[:, :, t], y_boot)
            betas[t, (j * n_features):(j * n_features) + n_features, k] = model.coef_
            mu[t, j, k] = model.intercept_
            mean_error[t, j, k] = np.mean(np.abs(y_boot - np.matmul(X_boot[:, :, t], model.coef_) - model.intercept_))

# ----------------------------------------------------------------------------------------------
# ---------------------------------- save results ----------------------------------------------

bootstrap_regr_dict = {'betas_ridge': betas, 'mu_ridge': mu, 'err_ridge': mean_error}

output_filename = "%s_vp_interaction_analysis_CI_bootstrap_%d.pkl" % (rat_name, n_bootstrap)
with open(os.path.join(results_directory, rat_name, output_filename), 'wb') as fp:
    pickle.dump(bootstrap_regr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

print('%s - INTERACTION analysis CI bootstrap computed and results stored - %s' % (rat_name, output_filename))


# ----------------------------------------------------------------------------------------------
# ---------------------------------- plot results ----------------------------------------------
# set confidence interval %
CI = 0.95
lim = int(CI * n_bootstrap)

# retrieve the neuron-state interaction coefficients
unit_idx = np.arange(0, n_units * n_features, n_features)
unit_state_idx = np.arange(1, n_units * n_features, n_features)
state_idx = np.arange(2, n_units * n_features, n_features)

# now we want to check that this unit-state weight is non zero
# betas_ridge_units = multiple_regr_dict[mouse_name]['betas_ridge'][:, unit_idx, 0]
betas_boot_unit_state = bootstrap_regr_dict['betas_ridge'][:, unit_state_idx, :]
betas_unit_state = bootstrap_regr_dict['betas_ridge'][:, unit_state_idx, 0]
# betas_ridge_state = multiple_regr_dict[mouse_name]['betas_ridge'][:, state_idx, 0]

# order the betas and select those within their 95% CI
betas_ordered = np.sort(betas_boot_unit_state, axis=2)
betas_lb = np.sign(betas_ordered[:, :, n_bootstrap-lim])
betas_ub = np.sign(betas_ordered[:, :, lim])
betas_nonzero = (betas_lb * betas_ub) > 0

# plot coefficients value - the nonzero ones
fig, ax = plt.subplots()
im2 = ax.imshow(betas_nonzero.transpose(1, 0)*betas_unit_state.transpose(1,0), cmap='cool', vmin=-2, vmax=2)
fig.colorbar(im2, ax=ax, location='bottom', orientation='horizontal')
fig.suptitle('nonzero betas, %s' % rat_name)
plt.show()
