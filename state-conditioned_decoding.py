"""
Script to run the state-conditioned decoding analysis.
The analysis should be run per rat.
It outputs the results of the analysis in a '.pkl' file, and plots the results.

Author: Laura Masaracchia
Email: laurama@cfin.au.dk
"""


import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from .utils.analysis import convert_vp_to_gamma, sort_trials_by_variables, group_trials_variable, run_class_balanced_decoding


# -------------------------------------------------------------------
# ------------------------ hyperparameters --------------------------
# choose rat name. Select from 'Buchanan', 'Stella', 'Mitt', 'Barat'
rat_name = 'Buchanan'
print(rat_name)

# Analysis hyperparameters
n_rep = 500
# minimum number for each decoding, i.e., per class, per state
min_trial_nbr = 5

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
# states into +1 and 0
gamma = gamma_vp_response[:, 0, :]

n_trials, n_units, n_time_points = units_density_response.shape

# ----------------------------------------------------------------------------------------------
# ---------------------- run regression per neuron and per time point --------------------------
y = success_label
X_units = units_density_response

# standardise units per time point and per trial, across neurons
X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

# initialize some measures
number_trials_state_class = np.empty(shape=(n_time_points,4))

w_state_acc = np.empty(shape=(n_time_points,))
c_state_acc = np.empty(shape=(n_time_points,))
k_w_state_acc = np.empty(shape=(n_rep,n_time_points, n_states))
k_c_state_acc = np.empty(shape=(n_rep,n_time_points, n_states))

for t in range(n_time_points):
    print('time point t = %d' % t)

    # for each time point get trials separated by behavior and state
    sorted_trials = sort_trials_by_variables(y[:, 0], gamma[:, 1, t])
    # outputed as:
    # 0A , 0B
    # 1A , 1B
    # group them by state
    state_cond_trials = group_trials_variable(sorted_trials, axis=1)

    # just for debug purposes, check the number of trials
    number_trials_state_class[t, :] = len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]), len(
        sorted_trials[1][1])

    # for each t store everything for each state and repetition
    X_test_t = {}
    y_true_t = {}
    y_pred_t = {}
    model_t = {}
    y_true_w_all = {}
    y_pred_w_all = {}
    y_true_c_all = {}
    y_pred_c_all = {}

    cross_pred_flag = np.zeros(shape=(n_states,))

    # for each state perform class-balanced decoding
    for k in range(n_states):

        X_state = X_units[state_cond_trials[k]]
        y_state = y[state_cond_trials[k]]
        # print('state %d'%k)
        # check for minimum amount of trials
        if min(np.sum(y_state==0), np.sum(y_state==1)) < min_trial_nbr:
            continue
        # if good, perform decoding
        else:
            cross_pred_flag[k] = 1
            k_w_state_acc[:,t,k], X_test_t[k], y_true_t[k], y_pred_t[k], model_t[k] = run_class_balanced_decoding(X_state,
                                                                                                                  y_state,
                                                                                                                  n_rep=n_rep,
                                                                                                                  return_all=True)
    # at this point we have all stored for both states
    # in the form of dictionary with one entry as k and one entry as b
    # see if you can compute cross decoding
    if np.sum(cross_pred_flag) == n_states:
        for k in range(n_states):

            y_true_w_all[k] = []
            y_pred_w_all[k] = []
            y_true_c_all[k] = []
            y_pred_c_all[k] = []

            for b in range(n_rep):
                model_state = model_t[k][b]
                x_test_cross = X_test_t[1 - k][b]
                y_test_cross = y_true_t[1 - k][b]

                y_pred_cross = model_state.predict(x_test_cross)
                k_c_state_acc[b,t,k] = accuracy_score(y_test_cross,y_pred_cross)

                # flatten outputs to compute unique accuracy
                y_true_w_all[k].extend(y_true_t[k][b])
                y_pred_w_all[k].extend(y_pred_t[k][b])
                y_true_c_all[k].extend(y_test_cross)
                y_pred_c_all[k].extend(y_pred_cross)

        # now compute within-state accuracy and cross-state accuracy
        within_all_true = np.concatenate((y_true_w_all[0],y_true_w_all[1]), axis=0)
        within_all_pred = np.concatenate((y_pred_w_all[0], y_pred_w_all[1]), axis=0)

        cross_all_true = np.concatenate((y_true_c_all[0],y_true_c_all[1]), axis=0)
        cross_all_pred = np.concatenate((y_pred_c_all[0], y_pred_c_all[1]), axis=0)

        w_state_acc[t] = accuracy_score(within_all_true,within_all_pred)
        c_state_acc[t] = accuracy_score(cross_all_true, cross_all_pred)


# ----------------------------------------------------------------------------------------------
# --------------------------------------- save results -----------------------------------------
state_cond_acc_res = {'all_within_state_acc': w_state_acc,
                      'each_within_state_acc': k_w_state_acc,
                      'all_cross_state_acc': c_state_acc,
                      'each_cross_state_acc': k_c_state_acc}

state_cond_dec_filename = '%s_class_balanced_state_cond_dec_rep%d.pkl' % (rat_name, n_rep)
with open(os.path.join(results_directory, rat_name, states_around_resp_filename), 'wb') as fp:
    pickle.dump(state_cond_acc_res, fp, protocol=pickle.HIGHEST_PROTOCOL)


print('Analysis finished and results stored, %s'%rat_name)
# ----------------------------------------------------------------------------------------------
# --------------------------------------- plot results -----------------------------------------

fig, ax = plt.subplots()
time_scale = np.arange(0, n_time_points)
ax.plot(time_scale, w_state_acc, color='orangered', label='per-state')
ax.plot(time_scale, c_state_acc, color='purple', label='cross-state')
ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
ax.vlines(501, 0, 1, 'k', linestyles='dashdot')
plt.legend()
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
ax.set_xlabel('t [s]')
ax.set_xlim([250, 1001])
ax.set_ylim([0.25, 1.05])
ax.set_ylabel('accuracy')
plt.title('state-conditioned decoding, trial outcome, %s' % rat_name)
plt.show()
