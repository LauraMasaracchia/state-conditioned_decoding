# data exploration of Fortin Lab data

import numpy as np
import os
import matplotlib
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.io
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from functions import *
import scipy.stats
#from Decoding_analysis.Fortin_data_analysis.utils.preproc import *
import sys

#----------------- randomised with real states --------------------------------------------
arg = int(sys.argv[1])
superchris_bad_trials = [0, 1, 4, 10, 137]

# modulations of
data_alignment = 'response'
decoding = 'success'  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap_perm = 10
n_bootstrap_state = 200
min_trial_nbr = 5
min_state_nbr = 20
n_permutations = 10001
n_perm_max = 15000

mouse_name_list = ['Buchanan','Stella','Mitt','Barat','Superchris']
mouse_name = mouse_name_list[arg]

print(mouse_name)
if mouse_name == 'Mitt':
    dirdiag = 10000000
else:
    dirdiag = 1000000000

n_states = 2
lags = 7
results_folder = os.path.join('/home/laura/Desktop/hippo_glhmm/', mouse_name)
gammas_response_name = 'pca_%s_vp_aligned_pokeout_orig_lag7_k2_dirdiag%d.mat' % (mouse_name, dirdiag)
mouse_folder = '/home/laura/Desktop/hippo_glhmm/%s' % mouse_name

info_file = '%s_trial_info.npy' % mouse_name.lower()
info_data = np.load(os.path.join(mouse_folder, info_file))

# NO LOCOMOTION! EXCLUDE CONTAMINATED DATA
movement_trials = scipy.io.loadmat(os.path.join(mouse_folder,'trials_to_discard.mat'))
movement_trials_response = movement_trials['trials_locomotion'][0] - 1
print('trials to exclude')
print(movement_trials_response)
info_data = np.delete(info_data, movement_trials_response, 0)

success_labels = info_data[:, 0]
inseq_labels = info_data[:, 1]
odor_labels = info_data[:, 3]

units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'


with open(os.path.join(mouse_folder, units_density_response_name), 'rb') as fp:
    units_density_response = pickle.load(fp)
units_density_response = np.delete(units_density_response, movement_trials_response, 0)

gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
states_around_response = gammas_trials_response['states_around_pokeout_orig']
# convert vp to gammas shape
gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
gamma_vp_response = np.delete(gamma_vp_response, movement_trials_response, 0)
n_states = gamma_vp_response.shape[1]

# -------------------------------- retrieve useful info -------------------------------------
n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
# choose kernel 1
n_time_points = 751
dk = 1

# ----------------------------- get the gammas shape from vpath -----------------------------

movement_filename = 'movement_trials_pokeout_aligned.mat'
movement_data = scipy.io.loadmat(os.path.join(results_folder, movement_filename))
XY_pos = movement_data['move_around_pokeout'][:,:,250:]
XY_vel = movement_data['vel_around_pokeout'][:,:,250:]
XY_pos = np.delete(XY_pos, movement_trials_response, 0)
XY_vel = np.delete(XY_vel, movement_trials_response, 0)
XY_pos[:, 0, :] = (XY_pos[:, 0, :] - np.mean(XY_pos[:, 0, :], 1)[:, np.newaxis]) / np.std(XY_pos[:, 0, :], 1)[:,
                                                                                   np.newaxis]
XY_pos[:, 1, :] = (XY_pos[:, 1, :] - np.mean(XY_pos[:, 1, :], 1)[:, np.newaxis]) / np.std(XY_pos[:, 1, :], 1)[:,
                                                                                   np.newaxis]

XY_vel[:, 0, :] = (XY_vel[:, 0, :] - np.mean(XY_vel[:, 0, :], 1)[:, np.newaxis]) / np.std(XY_vel[:, 0, :], 1)[:,
                                                                                   np.newaxis]
XY_vel[:, 1, :] = (XY_vel[:, 1, :] - np.mean(XY_vel[:, 1, :], 1)[:, np.newaxis]) / np.std(XY_vel[:, 1, :], 1)[:,
                                                                                   np.newaxis]
# --------------------------------------------------------------------------------------------------------
# -------------------------- analyses on gammas and units ---------------------------

x = units_density_response[:, :, 250:, dk]
x = (x - np.mean(x, 1)[:, np.newaxis,:]) / np.std(x, 1)[:, np.newaxis,:]
gamma = gamma_vp_response[:,:,250:]

X = np.zeros(shape=(x.shape))
y = success_labels

acc_standard = np.zeros(shape=(n_time_points, n_permutations))
acc_per_state = np.zeros(shape=(n_states, n_time_points, n_permutations))
acc_cross_state = np.zeros(shape=(n_states, n_time_points, n_permutations))
acc_per_state_general = np.empty(shape=(n_time_points,n_permutations))
acc_cross_state_general = np.empty(shape=(n_time_points,n_permutations))


permuted_states = np.zeros(shape=(n_time_points,n_permutations,n_trials))

for t in range(n_time_points):
    print('time point t = %d' % t)
    #deconfound first
    deconf_model = Ridge()
    deconf_model.fit(XY_pos[:, :, t], x[:, :, t])
    x_res = deconf_model.predict(XY_pos[:, :, t])
    X[:, :, t] = x[:, :, t] - x_res

    # check if this time point is worth computing at all
    state0_num = np.sum(gamma[:, 1, t]==0)
    state1_num = np.sum(gamma[:, 1, t]==1)
    if np.min([state0_num,state1_num])<min_state_nbr:
        print('not enough data in each state to do proper permutations, skip timepoint %d'%t)
        acc_standard[t,:] = np.nan
        acc_per_state_general[t,:] = np.nan
        acc_cross_state_general[t,:] = np.nan
        permuted_states[t,:,:] = np.nan

        acc_per_state[:,t,:] = np.nan
        acc_cross_state[:,t,:] = np.nan
        continue

    # otherwise we do the permutations

    n_bootstrap=n_bootstrap_state
    # do the permutations
    idx_success=np.where(y==1)[0]
    idx_fail = np.where(y==0)[0]

    for perm in range(n_permutations):
        if perm == 0:
            state_labels = gamma[:, 1, t].copy()  # real HMM
        else:
            state_labels = gamma[:, 1, t].copy()  # always start from original
            state_labels[idx_success] = np.random.permutation(state_labels[idx_success])
            state_labels[idx_fail] = np.random.permutation(state_labels[idx_fail])
            #state_labels = np.random.randint(0,2,size=(n_trials,))
            #state_labels = np.random.permutation(state_labels)
            n_bootstrap = n_bootstrap_perm

        sorted_trials = sort_trials_by_variables(y, state_labels)

        # check that there is a minimum number of trials in each condition
        smallest_nbr_available = min(len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]),len(sorted_trials[1][1]))
        if smallest_nbr_available < min_trial_nbr:
            print('not enough trials at this permutation, redo %d'%perm)
            continue

        # if we have enough trials, we go on with the computations
        # save states
        #permuted_states[t,perm,:] = gamma[:,1,t]

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

            # ------------------------------------- min nbr data --------------------------------------------
            # ------------------------- create train and test data for this fold -----------------------
            # ------------------------- subsample to min nbr available -----------------------
            # find minimum number of trials in this state to train
            #min_nbr_per_class_state0 = min(len(sorted_trials[0][0]), len(sorted_trials[1][0]))
            min_nbr_per_class_state0 = smallest_nbr_available
            train_nbr_per_class_state0 = int(0.8 * min_nbr_per_class_state0)
            test_nbr_per_class_state0 = min_nbr_per_class_state0 - train_nbr_per_class_state0

            train_idx_class1_state0 = np.array(sorted_trials[0][0])[:train_nbr_per_class_state0]
            train_idx_class2_state0 = np.array(sorted_trials[1][0])[:train_nbr_per_class_state0]
            test_idx_class1_state0 = np.array(sorted_trials[0][0])[train_nbr_per_class_state0:train_nbr_per_class_state0+test_nbr_per_class_state0]
            test_idx_class2_state0 = np.array(sorted_trials[1][0])[train_nbr_per_class_state0:train_nbr_per_class_state0+test_nbr_per_class_state0]

            tot_train_trials_state0 = np.concatenate((train_idx_class1_state0, train_idx_class2_state0), 0)
            tot_test_trials_state0 = np.concatenate((test_idx_class1_state0, test_idx_class2_state0), 0)

            # find minimum number of trials in this state to train
            #min_nbr_per_class_state1 = min(len(sorted_trials[0][1]), len(sorted_trials[1][1]))
            min_nbr_per_class_state1 = smallest_nbr_available
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

            #model_standard = RidgeClassifier(alpha=0.001)
            model_state0 = RidgeClassifierCV(alphas=np.array([0.001,0.01,0.1,1,10,100,500]))
            model_state1 = RidgeClassifierCV(alphas=np.array([0.001,0.01,0.1,1,10,100,500]))

            # train standard model
            #model_standard.fit(X[tot_train_trials,:,t],y[tot_train_trials])

            # test standard model on all
            #y_standard_pred.extend(model_standard.predict(X[tot_test_trials,:,t]))
            #y_standard_true.extend(y[tot_test_trials])

            # train per state models
            model_state0.fit(X[tot_train_trials_state0, :, t], y[tot_train_trials_state0])
            model_state1.fit(X[tot_train_trials_state1, :, t], y[tot_train_trials_state1])

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
        #acc_standard[t, perm] = accuracy_score(y_standard_true, y_standard_pred)
        acc_per_state_general[t,perm] = accuracy_score(y_state_true,y_state_pred)
        acc_cross_state_general[t,perm] = accuracy_score(y_state_true,y_cross_pred)
        #acc_per_state[0, t, perm] = accuracy_score(y_state0_true, y_state0_pred_state0)
        #acc_per_state[1, t, perm] = accuracy_score(y_state1_true, y_state1_pred_state1)
        #acc_cross_state[1, t, perm] = accuracy_score(y_state1_true, y_state0_pred_state1)
        #acc_cross_state[0, t, perm] = accuracy_score(y_state0_true, y_state1_pred_state0)


output_name = "NO_LOCO_stand_rand_deco_pos_subsampled_10kperm_%s_units_per_state_decoding_success-nbootstrap_%d_state.pkl" % (model_name, n_bootstrap_state)

result_perm_dict= {#"acc_standard": acc_standard,
                   #'acc_per_state': acc_per_state,
                   #'acc_cross_state': acc_cross_state,
                   'acc_per_state_general': acc_per_state_general,
                   'acc_cross_state_general':acc_cross_state_general}
                   #'permuted_states': permuted_states}

with open(os.path.join(results_folder, output_name), 'wb') as fp:
    pickle.dump(result_perm_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

print('%s - Analysis finished and results stored - %s' % (mouse_name, output_name))




