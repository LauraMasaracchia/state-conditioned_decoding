"""
Script to run the statistical analysis on the state-conditioned decoding.
The analysis should be run per rat.

This script performs:
 1. State-conditioned decoding on random states, 10000 permutations
 2. p-value computation
 3. window-based correction of p-values
 4. Plotting of final results

It outputs the results of the analysis in a '.pkl' file, and plots the results.

Author: Laura Masaracchia
Email: laurama@cfin.au.dk
"""

import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
import sys
from .utils.analysis import *
from glhmm import statistics


#----------------- randomised with real states --------------------------------------------
arg = int(sys.argv[1])
superchris_bad_trials = [0, 1, 4, 10, 137]

# modulations of
data_alignment = 'response'
decoding = 'success'  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap_perm = 10
n_bootstrap_state = 250
min_trial_nbr = 5
min_state_nbr = 20
n_permutations = 10001
n_perm_max = 15000

mouse_name_list = ['Buchanan','Stella','Mitt','Barat','Superchris']
mouse_name = mouse_name_list[arg]

print(mouse_name)
if mouse_name == 'Mitt':
    gamma_directory = 'gamma_k2_l7_d10m'
    dirdiag = 10000000
else:
    gamma_directory = 'gamma_k2_l7_d1bn'
    dirdiag = 1000000000

n_states = 2
lags = 7
results_folder = os.path.join('/home/laura/Desktop/hippo_glhmm/', mouse_name, gamma_directory)
gammas_response_name = 'pca_%s_vp_aligned_pokeout_lag7_k2_dirdiag%d.mat' % (mouse_name, dirdiag)
mouse_folder = '/home/laura/Desktop/hippo_glhmm/%s' % mouse_name

info_file = '%s_trial_info.npy' % mouse_name.lower()
info_data = np.load(os.path.join(mouse_folder, info_file))

if mouse_name == 'Superchris':
    info_data = np.delete(info_data, superchris_bad_trials, 0)
success_labels = info_data[:, 0]
inseq_labels = info_data[:, 1]
odor_labels = info_data[:, 3]

units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

with open(os.path.join(mouse_folder, units_density_response_name), 'rb') as fp:
    units_density_response = pickle.load(fp)

gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
states_around_response = gammas_trials_response['states_around_pokeout']

# -------------------------------- retrieve useful info -------------------------------------
n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
# choose kernel 1
n_time_points = 751
dk = 1

# ----------------------------- get the gammas shape from vpath -----------------------------
# convert vp to gammas shape
gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
n_states = gamma_vp_response.shape[1]

# --------------------------------------------------------------------------------------------------------
# -------------------------- analyses on gammas and units ---------------------------

X = units_density_response[:, :, 250:, dk]
X = (X - np.mean(X, 1)[:, np.newaxis,:]) / np.std(X, 1)[:, np.newaxis,:]
gamma = gamma_vp_response[:,:,250:]

y = success_labels

acc_standard = np.zeros(shape=(n_time_points, n_permutations))
acc_per_state = np.zeros(shape=(n_states, n_time_points, n_permutations))
acc_cross_state = np.zeros(shape=(n_states, n_time_points, n_permutations))
acc_per_state_general = np.zeros(shape=(n_time_points,n_permutations))
acc_cross_state_general = np.zeros(shape=(n_time_points,n_permutations))
acc_tot_avg = np.zeros(shape=(n_time_points,  n_permutations))

permuted_states = np.zeros(shape=(n_time_points,n_permutations,n_trials))

for t in range(n_time_points):
    print('time point t = %d' % t)

    # check if this time point is worth computing at all
    state0_num = np.sum(gamma[:, 1, t]==0)
    state1_num = np.sum(gamma[:, 1, t]==1)
    if np.min([state0_num,state1_num])<min_state_nbr:
        print('not enough data in each state to do proper permutations, skip timepoint %d'%t)
        acc_standard[t,:] = np.nan
        acc_per_state_general[t,:] = np.nan
        acc_cross_state_general[t,:] = np.nan
        permuted_states[t,:,:] = np.nan
        acc_tot_avg[t,:] = np.nan
        acc_per_state[:,t,:] = np.nan
        acc_cross_state[:,t,:] = np.nan
        continue

    # otherwise we do the permutations
    perm_tot = 0
    perm = 0
    n_bootstrap=n_bootstrap_state
    # do the permutations
    while perm_tot < n_perm_max and perm < n_permutations:
        # shuffle
        # compute the right one the first time, permute the next 10k times
        if perm>=1:
            np.random.shuffle(gamma[:, 1, t])
            n_bootstrap = n_bootstrap_perm

        sorted_trials = sort_trials_by_variables(y, gamma[:,1,t])

        # check that there is a minimum number of trials in each condition
        if min(len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]),len(sorted_trials[1][1])) < min_trial_nbr:
            print('not enough trials at this permutation, redo %d'%perm)
            perm_tot +=1
            continue

        # if we have enough trials, we go on with the computations
        # save states
        permuted_states[t,perm,:] = gamma[:,1,t]

        y_standard_true = []
        y_standard_pred = []
        y_avg_pred = []
        y_state_true = []
        y_state_pred = []
        y_cross_pred = []
        # state-related
        y_state0_true = []
        y_state1_true = []
        y_state0_pred_state0 = []
        y_state1_pred_state1 = []
        y_state0_pred_state1 = []
        y_state1_pred_state0 = []

        for f in range(n_bootstrap):

            # shuffle the data and create the train/test split for all groups
            np.random.shuffle(sorted_trials[0][0])
            np.random.shuffle(sorted_trials[1][0])
            np.random.shuffle(sorted_trials[0][1])
            np.random.shuffle(sorted_trials[1][1])

            # ------------------------------------- STATE 0 --------------------------------------------
            # ------------------------- create train and test data for this fold -----------------------
            # find minimum number of trials in this state to train
            min_nbr_per_class_state0 = min(len(sorted_trials[0][0]), len(sorted_trials[1][0]))
            train_nbr_per_class_state0 = int(0.8 * min_nbr_per_class_state0)
            test_nbr_per_class_state0 = min_nbr_per_class_state0 - train_nbr_per_class_state0

            train_idx_class1_state0 = np.array(sorted_trials[0][0])[:train_nbr_per_class_state0]
            train_idx_class2_state0 = np.array(sorted_trials[1][0])[:train_nbr_per_class_state0]
            test_idx_class1_state0 = np.array(sorted_trials[0][0])[train_nbr_per_class_state0:train_nbr_per_class_state0+test_nbr_per_class_state0]
            test_idx_class2_state0 = np.array(sorted_trials[1][0])[train_nbr_per_class_state0:train_nbr_per_class_state0+test_nbr_per_class_state0]

            tot_train_trials_state0 = np.concatenate((train_idx_class1_state0, train_idx_class2_state0), 0)
            tot_test_trials_state0 = np.concatenate((test_idx_class1_state0, test_idx_class2_state0), 0)

            # find minimum number of trials in this state to train
            min_nbr_per_class_state1 = min(len(sorted_trials[0][1]), len(sorted_trials[1][1]))
            train_nbr_per_class_state1 = int(0.8 * min_nbr_per_class_state1)
            test_nbr_per_class_state1 = min_nbr_per_class_state1 - train_nbr_per_class_state1

            train_idx_class1_state1 = np.array(sorted_trials[0][1])[:train_nbr_per_class_state1]
            train_idx_class2_state1 = np.array(sorted_trials[1][1])[:train_nbr_per_class_state1]
            test_idx_class1_state1 = np.array(sorted_trials[0][1])[
                                             train_nbr_per_class_state1:train_nbr_per_class_state1 + test_nbr_per_class_state1]
            test_idx_class2_state1 = np.array(sorted_trials[1][1])[
                                             train_nbr_per_class_state1:train_nbr_per_class_state1 + test_nbr_per_class_state1]

            tot_train_trials_state1 = np.concatenate((train_idx_class1_state1, train_idx_class2_state1), 0)
            tot_test_trials_state1 = np.concatenate((test_idx_class1_state1, test_idx_class2_state1), 0)

            tot_train_trials = np.concatenate((tot_train_trials_state0,tot_train_trials_state1),0)
            tot_test_trials = np.concatenate((tot_test_trials_state0, tot_test_trials_state1), 0)
            np.random.shuffle(tot_train_trials)

            model_standard = RidgeClassifierCV(alphas=np.array([0.01,0.1,1.0,10,100]))
            model_state0 = RidgeClassifierCV(alphas=np.array([0.1,1,10,100]))
            model_state1 = RidgeClassifierCV(alphas=np.array([0.1,1,10,100]))
            model_avg = RidgeClassifier()
            # train standard model
            model_standard.fit(X[tot_train_trials,:,t],y[tot_train_trials])
            model_avg.fit(X[tot_train_trials,:,t],y[tot_train_trials])

            # test standard model on all
            y_standard_pred.extend(model_standard.predict(X[tot_test_trials,:,t]))
            y_standard_true.extend(y[tot_test_trials])

            # train per state models
            model_state0.fit(X[tot_train_trials_state0, :, t], y[tot_train_trials_state0])
            model_state1.fit(X[tot_train_trials_state1, :, t], y[tot_train_trials_state1])

            # check the avg model
            model_avg.coef_ = (model_state1.coef_ + model_state0.coef_) / 2
            model_avg.intercept_ = (model_state1.intercept_ + model_state0.intercept_) / 2

            y_avg_pred.extend(model_avg.predict(X[tot_test_trials, :, t]))

            # test model states within a state
            y_state0_pred_state0.extend(model_state0.predict(X[tot_test_trials_state0,:,t]))
            y_state0_true.extend(y[tot_test_trials_state0])

            y_state1_pred_state1.extend(model_state1.predict(X[tot_test_trials_state1,:,t]))
            y_state1_true.extend(y[tot_test_trials_state1])

            y_state_true.extend(np.concatenate((y_state0_true,y_state1_true),axis=0))
            y_state_pred.extend(np.concatenate((y_state0_pred_state0,y_state1_pred_state1),axis=0))

            # test state models across state
            y_state0_pred_state1.extend(model_state0.predict(X[tot_test_trials_state1,:,t]))
            y_state1_pred_state0.extend(model_state1.predict(X[tot_test_trials_state0,:,t]))

            y_cross_pred.extend(np.concatenate((y_state1_pred_state0,y_state0_pred_state1),axis=0))

        # accuracy here
        acc_standard[t, perm] = accuracy_score(y_standard_true, y_standard_pred)
        acc_tot_avg[t, perm] = accuracy_score(y_standard_true, y_avg_pred)
        acc_per_state_general[t,perm] = accuracy_score(y_state_true,y_state_pred)
        acc_cross_state_general[t,perm] = accuracy_score(y_state_true,y_cross_pred)
        acc_per_state[0, t, perm] = accuracy_score(y_state0_true, y_state0_pred_state0)
        acc_per_state[1, t, perm] = accuracy_score(y_state1_true, y_state1_pred_state1)
        acc_cross_state[1, t, perm] = accuracy_score(y_state1_true, y_state0_pred_state1)
        acc_cross_state[0, t, perm] = accuracy_score(y_state0_true, y_state1_pred_state0)

        perm +=1
        perm_tot +=1

output_name = "standardised_randomised_10kperm_%s_units_per_state_decoding_success-nbootstrap_%d_state.pkl" % (model_name, n_bootstrap)

result_perm_dict= {"acc_standard": acc_standard,
                   'acc_per_state': acc_per_state,
                   'acc_cross_state': acc_cross_state,
                   'acc_tot_avg': acc_tot_avg,
                   'acc_per_state_general': acc_per_state_general,
                   'acc_cross_state_general':acc_cross_state_general,
                   'permuted_states': permuted_states}

with open(os.path.join(results_folder, output_name), 'wb') as fp:
    pickle.dump(result_perm_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

print('%s - Analysis finished and results stored - %s' % (mouse_name, output_name))


# ---------------------------------------------------------------------------------------------------
# ------------------------------ load all results, compute pvalue and plot --------------------------
# ---------------------------------------------------------------------------------------------------


result_perm_dict = {}
for mouse_name in mouse_name_list:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        # gammas
        n_states = 2
        lags = 7
        dirdiag = 10000000
    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        # gammas
        n_states = 2
        lags = 7
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, gamma_directory)
    output_name = "standardised_randomised_10kperm_%s_units_per_state_decoding_%s-nbootstrap_%d_state.pkl" % (model_name, 'success', n_bootstrap)
    if mouse_name == 'Mitt':
        output_name = "standardised_randomised_1kperm_%s_units_per_state_decoding_%s-nbootstrap_%d_state.pkl" % (
        model_name, 'success', n_bootstrap)
    with open(os.path.join(results_folder,output_name), 'rb') as fp:
        result_perm_dict[mouse_name] = pickle.load(fp)

    ref_acc = result_perm_dict[mouse_name]['acc_standard']
    perm_count = 10000 - np.sum(ref_acc==0,1)

    fig = plt.figure()
    plt.plot(np.arange(751),perm_count)
    plt.title('valid perms, %s'%mouse_name)
    plt.show()


    fig, ax = plt.subplots()
    time_scale = np.arange(0, n_time_points)
    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_standard'][:,0], color='navy', label='standard')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_standard'][:,1:1000], axis=1), color='steelblue', label='stand_perm')

    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_tot_avg'][:,0], color='blue', label='avg hmm')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_tot_avg'][:,1:1000], axis=1), color='cyan', label='avg perm')

    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_per_state'][0,:,0], color='crimson', label='HMM_state1')
    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_per_state'][1,:,0], color='orange', label='HMM_state2')

    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state'][0,:,1:1000],axis=1), color='red', label='perm_state1')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state'][1,:,1:1000],axis=1), color='yellow', label='perm_state2')

    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_per_state_general'][:,0], color='darkorange', label='HMM_per_state')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:,1:1000],axis=1), color='orchid', label='perm_per_state')

    ax.plot(time_scale, result_perm_dict[mouse_name]['acc_cross_state_general'][:,0], color='purple', label='HMM_cross_state')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:,1:1000],axis=1), color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    plt.legend()
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0,751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')
    plt.title('Within and cross decoding, decoding success, %s' % mouse_name)
    plt.show()

n_time_points = 751
win_len = 20
win_skip = 10
time_scale = np.arange(n_time_points)
for mouse_name in ['Buchanan', 'Stella', 'Mitt']:
    n_time_points = 751
    time_scale = np.arange(n_time_points)

    result_perm_dict[mouse_name]['acc_per_state_general'][
        result_perm_dict[mouse_name]['acc_per_state_general'] == 0] = np.nan
    result_perm_dict[mouse_name]['acc_cross_state_general'][
        result_perm_dict[mouse_name]['acc_cross_state_general'] == 0] = np.nan

    ref_acc = result_perm_dict[mouse_name]['acc_standard']


    if mouse_name == 'Buchanan':
        min_perm_nbr = 2000
    else:
        min_perm_nbr = 1000

    if mouse_name == 'Mitt':
        HMM_diff_curve = result_perm_dict[mouse_name]['acc_per_state_general'][:, 0] - \
                         result_perm_dict[mouse_name]['acc_cross_state_general'][:, 0]
        hmm_per_state = result_perm_dict[mouse_name]['acc_per_state_general'][:, 0]
        hmm_cross_state = result_perm_dict[mouse_name]['acc_cross_state_general'][:, 0]
        time_points_center = np.arange(200 - int(win_len / 2), int(680 - win_len / 2), win_skip)
        new_pvalues = np.ones(shape=(len(time_points_center),))

    else:
        HMM_diff_curve = result_perm_dict[mouse_name]['acc_per_state_general'][:, 0] - result_perm_dict[mouse_name]['acc_cross_state_general'][:, 0]
        hmm_per_state = result_perm_dict[mouse_name]['acc_per_state_general'][:, 0]
        hmm_cross_state = result_perm_dict[mouse_name]['acc_cross_state_general'][:, 0]
        time_points_center = np.arange(250 - int(win_len / 2), int(n_time_points - win_len / 2), win_skip)
        new_pvalues = np.ones(shape=(len(time_points_center),))
        # here fill in the missing permutations

    random_diff_curve = result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:min_perm_nbr + 1] - \
                            result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:min_perm_nbr + 1]

    i = 0
    for t in time_points_center:
        avg_win_hmm = np.mean(HMM_diff_curve[t - int(win_len / 2): t + int(win_len / 2)], 0)
        win_rnd = np.ones(shape=(win_len, 10000))
        win_rnd[:, :min_perm_nbr] = random_diff_curve[t - int(win_len / 2): t + int(win_len / 2), :]
        for k in range(min_perm_nbr, 10000):
            win_indices = np.random.randint(0, min_perm_nbr - 1, win_len)
            win_rnd[:, k] = random_diff_curve[np.arange(win_len), win_indices]

        avg_win_rnd = np.mean(win_rnd, 0)
        new_pvalues[i] = np.sum(avg_win_hmm - avg_win_rnd < 0) / 10000
        i += 1
    #new_pvalues[new_pvalues == 0] = np.nan
    pval_corrected_stand, rejected_corrected = statistics.pval_correction(new_pvalues, method='fdr_bh', alpha=0.05)

    fig, ax = plt.subplots()

    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(time_points_center, pval_corrected_stand < 0.05, '*g')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
            color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
            color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    #ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('Within-state decoding minus cross-state decoding, %s' % mouse_name)
    plt.show()
