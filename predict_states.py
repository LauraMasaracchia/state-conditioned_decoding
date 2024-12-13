"""
This script runs the analyses used to infer the connection between the neurons and the states, i.e. whether different
states recruited different neurons within the population.
It uses neural activity, state time courses, and behavioural information.
It performs two main analyses:
1. average neural activity per neuron is computed under each state,
2. state time courses are predicted by the trial-by-trial standardised neural activity, per time point.

Author: Laura Masaracchia, laurama@cfin.au.dk
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import scipy.stats
from .utils.analysis import convert_vp_to_gamma, run_class_balanced_decoding, compute_CI


# -------------------------------------------------------------------
# ------------------------ hyperparameters --------------------------
# choose rat name. Select from 'Buchanan', 'Stella', 'Mitt', 'Barat'
rat_name = 'Buchanan'
print(rat_name)

# Analysis hyperparameters
n_rep = 500


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
    X_units = pickle.load(fp)


poke_out_time = behav_info['poke_out_time']
# success label is with +1 if success, -1 if fail
success_label = behav_info['success_label']

states_around_response = states_response_dict['VP']
gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
# get states into +1 and -1
gamma = gamma_vp_response[:, 0, :] * 2.0 - 1.0

n_trials, n_units, n_time_points = X_units.shape


# ------------------------------------------------------------------------------------------------
# ------------------ compute avg firing rate of each neuron per state ----------------------------
gamma1t = np.where(gamma[:, 0] == 1)[0]
gamma2t = np.where(gamma[:, 0] == 0)[0]
Xstate1 = X_units[gamma1t, :, 0]
Xstate2 = X_units[gamma2t, :, 0]

for t in range(1,n_time_points):
    gamma1t = np.where(gamma[:, t] == 1)[0]
    gamma2t = np.where(gamma[:, t] == 0)[0]

    Xstate1 = np.concatenate([Xstate1, X_units[gamma1t, :, t]], axis=0)
    Xstate2 = np.concatenate([Xstate2, X_units[gamma2t, :, t]], axis=0)


# welch's t test
stat = scipy.stats.ttest_ind(Xstate1, Xstate2, axis=0, equal_var=False)

# plot
fig = plt.figure()
plt.bar(np.arange(n_units)-0.2,np.mean(Xstate1,0),width=0.4,color='crimson')
plt.bar(np.arange(n_units)+0.2,np.mean(Xstate2,0),width=0.4,color='orange')
yerr1 = np.std(Xstate1,0)
yerr2 = np.std(Xstate2,0)
plt.errorbar(np.arange(n_units)-0.2,np.mean(Xstate1,0), yerr1, fmt='.', zorder=0, elinewidth=0.2,color='k', alpha=0.7)
plt.errorbar(np.arange(n_units)+0.2, np.mean(Xstate2, 0), yerr2, fmt='.', zorder=0, elinewidth=0.2, color='k',alpha=0.7)
plt.plot(np.arange(n_units), 1 - (stat.pvalue < 0.02), '*k')
plt.xlabel('neurons')
plt.ylabel('avg fire rate')
plt.title('avg fire rate per state, %s'%rat_name)
plt.show()


# -----------------------------------------------------------------------------------------------------------
# ------------------ predict states activation from standardised neural activity ----------------------------

# standardise units per time point and per trial, across neurons
X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

accuracy_state_pred = np.empty(n_time_points,n_rep)

for t in range(n_time_points):
    y_t = gamma[:, t]
    X_t = X_units[:, :, t]
    accuracy_state_pred[t,:] = run_class_balanced_decoding(X=X_t,
                                                           y=y_t,
                                                           n_rep=n_rep,
                                                           return_coef=False)


# compute confidence interval
mean_acc, interval_acc = compute_CI(accuracy_state_pred, axis=1)


# -----------------------------------------------------------------------------------
# plot results
# plot decoding results overposed to trial bars
fig, ax = plt.subplots()
time_scale = np.arange(0, n_time_points)
ax.plot(np.arange(n_time_points), mean_acc, color='navy')
ax.fill_between(np.arange(n_time_points),interval_acc[0], interval_acc[1], alpha=0.5,color='navy')
ax.plot(np.arange(n_time_points), np.ones(shape=(n_time_points,)) * 0.5, '--', color='coral')
ax.vlines(500, 0, 1, 'k', linestyles='dashdot')
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['-2','-1', '0', '1', '2'])
ax.set_ylim([0.4, 1.1])
ax.set_ylabel('accuracy')
plt.title('%s prediction of state from standardised neural activity' % rat_name)
plt.show()


# -----------------------------------------------------------------------------------
# save results

pred_acc_filename = '%s_predict_states_from_units_rep%d.pkl'%(rat_name,n_rep)
with open(os.path.join(results_directory, rat_name, pred_acc_filename), 'wb') as fp:
    pickle.dump(accuracy_state_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)

