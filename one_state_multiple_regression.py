
# data exploration of Fortin Lab data
from elephant.statistics import instantaneous_rate
from quantities import ms
from neo import SpikeTrain
from elephant.kernels import GaussianKernel
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier, ElasticNetCV, LinearRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import RidgeCV, Lasso
# if on local
from Decoding_analysis.Fortin_data_analysis.utils.functions import *
# if on Morpheus
import sys
#from functions import *

# ------------------- use only one state for the regression -----------------------
#  check the regression and performance per unit

# if on Morpheus
#arg = int(sys.argv[1])

# HMM gammas specifications
# gammas at response have -500 and +500 aroun

mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']
# if on Morpheus
#mouse_name = mouse_name_list[arg]

for mouse_name in mouse_name_list:
# if on Morpheus
#for mouse_name in [mouse_name]:
    print(mouse_name)

    # review
    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
    info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/', mouse_name)

    info_file = '%s_trial_info.npy'%mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))

    movement_trials = scipy.io.loadmat(
    os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses', 'trials_to_discard.mat'))
    movement_trials_response = movement_trials['trials_locomotion'][0] - 1
    info_data = np.delete(info_data, movement_trials_response, 0)

    success_labels = info_data[:,0]
    inseq_labels = info_data[:,1]
    odor_labels = info_data[:,3]

    units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s'%mouse_name
    units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

    with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
        units_density_response = pickle.load(fp)

    units_density_response = np.delete(units_density_response, movement_trials_response, 0)

    # -------------------------------- retrieve useful info -------------------------------------
    n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
    # choose kernel 1
    dk = 1
    n_time_points = 751

    y = success_labels

    X_units = units_density_response[:, :, 250:, dk]
    X_dec = np.zeros(shape=X_units.shape)
    movement_filename = 'movement_trials_pokeout_aligned.mat'

    movement_data = scipy.io.loadmat(os.path.join(results_folder, movement_filename))

    XY_pos = movement_data['move_around_pokeout'][:, :, 250:]
    XY_vel = movement_data['vel_around_pokeout'][:, :, 250:]
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

    # either normalize the units this way or per state
    X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

    # normalize the units per state
    # run it with standardised units activity per state
    for t in range(n_time_points):
    # regress out movement
        deconf_model = Ridge()
        deconf_model.fit(XY_pos[:, :, t], X_units[:, :, t])
        x_res = deconf_model.predict(XY_pos[:, :, t])
        X_dec[:, :, t] = X_units[:, :, t] - x_res

    betas_ridge = np.empty(shape=(n_time_points, n_units))
    mu_ridge = np.empty(shape=(n_time_points, n_units))

    mean_error_ridge = np.empty(shape=(n_time_points, n_units))
    #accuracy = np.empty(shape=(n_time_points, n_units))

    for j in range(n_units):
        # took away gamma because unable to allocate that much memory
        X = X_dec[:, j, :]
        # first run the unpermuted regression
        for t in range(n_time_points):
            model2 = Ridge(alpha=0.01)
            model2.fit(X[:, t].reshape(-1,1), y)
            betas_ridge[t, j] = model2.coef_
            mu_ridge[t, j] = model2.intercept_
            mean_error_ridge[t, j] = np.mean(np.abs(y - (X[:, t]* model2.coef_)))
            #accuracy[t, j] = np.mean(y == (X[:, t]* model2.coef_) + model2.intercept_)

    fig, ax = plt.subplots()  # (nrows=3)
    ax.plot(np.arange(n_time_points), np.mean(mean_error_ridge, 1), 'navy', label='MAE')
    #ax.plot(np.arange(n_time_points), np.mean(accuracy, 1), 'steelblue',label='accuracy')
    ax.set_xticks([0, 250, 500, 750])
    plt.legend()
    plt.xlabel('time')
    plt.title('average prediction error %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()  # (nrows=3)
    im = ax.imshow(betas_ridge.transpose(), cmap='coolwarm')
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 250, 500, 750])
    plt.xlabel('time')
    plt.ylabel('units')
    plt.title('weights %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()  # (nrows=3)
    im = ax.imshow(mu_ridge.transpose(), cmap='coolwarm')
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 250, 500, 750])
    plt.xlabel('time')
    plt.ylabel('units')
    plt.title('intercepts %s' % mouse_name)
    plt.show()



# HMM gammas specifications
# gammas at response have -500 and +500 aroun

n_perms = 20

mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']
# if on Morpheus
#mouse_name = mouse_name_list[arg]

for mouse_name in mouse_name_list:
# if on Morpheus
#for mouse_name in [mouse_name]:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        dirdiag = 10000000
        lags = 7
        n_states = 2

    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        dirdiag = 1000000000
        lags = 7
        n_states = 2

    # review
    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
    info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                                mouse_name)
    # if on local
    #results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,gamma_directory)
    #info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',mouse_name)
    # if on Morpheus
    #results_folder = os.path.join('/home/laura/Desktop/hippo_glhmm/', mouse_name, gamma_directory)
    #info_data_folder = os.path.join('/home/laura/Desktop/hippo_glhmm/', mouse_name)

    #gammas_response_name = 'pca_%s_gammas_aligned_pokeout_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)
    gammas_response_name = 'pca_%s_vp_aligned_pokeout_orig_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)

    info_file = '%s_trial_info.npy'%mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))

    movement_trials = scipy.io.loadmat(
    os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses',
                 'trials_to_discard.mat'))
    movement_trials_response = movement_trials['trials_locomotion'][0] - 1
    info_data = np.delete(info_data, movement_trials_response, 0)

    success_labels = info_data[:,0]
    inseq_labels = info_data[:,1]
    odor_labels = info_data[:,3]

    units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s'%mouse_name
    units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

    with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
        units_density_response = pickle.load(fp)

    units_density_response = np.delete(units_density_response, movement_trials_response, 0)

    gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
    # if with prob
    #states_around_response = gammas_trials_response['states_around_pokeout'].transpose(0,2,1)
    # if with vp
    states_around_response = gammas_trials_response['states_around_pokeout_orig']
    states_around_response = np.delete(states_around_response, movement_trials_response,0)
    gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)

    # -------------------------------- retrieve useful info -------------------------------------
    n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
    # choose kernel 1
    dk = 1
    n_time_points = 751

    # if with prob
    # gamma = states_around_response[:, 0, 250:]
    gamma = gamma_vp_response[:, 0, 250:] * 2.0 - 1.0
    print(np.unique(gamma))
    y = success_labels * 2.0 - 1.0
    idx_success = np.where(y == 1)[0]
    idx_fail = np.where(y == 0)[0]

    X_units = units_density_response[:, :, 250:, dk]
    X_dec = np.zeros(shape=X_units.shape)
    movement_filename = 'movement_trials_pokeout_aligned.mat'

    movement_data = scipy.io.loadmat(os.path.join(results_folder, movement_filename))
    XY_pos = movement_data['move_around_pokeout'][:, :, 250:]
    XY_vel = movement_data['vel_around_pokeout'][:, :, 250:]
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

    # either normalize the units this way or per state
    X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

    # normalize the units per state
    # run it with standardised units activity per state
    for t in range(n_time_points):
    # regress out movement
        deconf_model = Ridge()
        deconf_model.fit(XY_pos[:, :, t], X_units[:, :, t])
        x_res = deconf_model.predict(XY_pos[:, :, t])
        X_dec[:, :, t] = X_units[:, :, t] - x_res

    # first compute it with no perms, take it as first to compare
    # at this point I should have my X and my Y
    # start with regression with bootstrap at each timepoint
    n_features = 3
    #betas = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
    #mu = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    betas_ridge = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
    mu_ridge = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    #betas_lasso = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
    #mu_lasso = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    mean_error_ridge = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    #accuracy = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    #mean_error_lasso = np.empty(shape=(n_time_points, n_units, n_perms + 1))

    units_mult_gamma1 = X_dec * gamma[:, np.newaxis, :]
    #fakegammas=np.zeros(shape=(n_trials,1,n_time_points))
    # the first one is the basic statistic: run the regression per neuron, no swapping
    for j in range(n_units):
        # took away gamma because unable to allocate that much memory
        X = np.concatenate((X_dec[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], gamma[:, np.newaxis, :]), axis=1)
        # first run the unpermuted regression
        for t in range(n_time_points):
            # first the original, unshuffled
            #model = RidgeClassifier(alpha=0.001)
            #model.fit(X[:, :, t], y)
            #betas[t, (j * n_features):(j * n_features) + n_features, 0] = model.coef_
            #mu[t, j, 0] = model.intercept_
            #pred = model.predict(X[:,:,t])
            #accuracy[t,j,0] = accuracy_score(y,pred)

            model2 = Ridge()
            model2.fit(X[:, :, t], y)
            betas_ridge[t, (j * n_features):(j * n_features) + n_features, 0] = model2.coef_
            mu_ridge[t, j, 0] = model2.intercept_
            mean_error_ridge[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model2.coef_) - model2.intercept_))

            #model3 = Lasso()
            #model3.fit(X[:, :, t], y)
            #betas_lasso[t, (j * n_features):(j * n_features) + n_features, 0] = model3.coef_
            #mu_lasso[t, j, 0] = model3.intercept_
            #mean_error_lasso[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model3.coef_) - model3.intercept_))

    # then we do with the permutations swapping gammas, the same index for all neurons and for all time points.
    # swap for each iteration

    for k in range(1, n_perms + 1):
        print(k)
        # for permutations
        swapped_gamma = gamma.copy()
        swap_idx = np.random.choice(np.arange(n_trials), int(n_trials/2),replace=False)
        swapped_gamma[swap_idx, :] = -gamma[swap_idx,:]
        #swapped_gamma[idx_success] = np.random.permutation(swapped_gamma[idx_success])
        #swapped_gamma[idx_fail] = np.random.permutation(swapped_gamma[idx_fail])
        #swapped_gamma = np.random.permutation(swapped_gamma)
        #swapped_gamma = np.random.randint(0, 2, size=(n_trials,n_time_points)) * 2.0 - 1.0

        units_mult_gamma1= X_dec * swapped_gamma[:,np.newaxis,:]

        for j in range(n_units):
            # resample indices with replacement
            X = np.concatenate((X_dec[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], swapped_gamma[:,np.newaxis,:]), axis=1)
            for t in range(n_time_points):
                #model = RidgeClassifier(alpha=0.001)
                #model.fit(X[:, :, t], y)
                #betas[t, (j * n_features):(j * n_features) + n_features, k] = model.coef_
                #mu[t, j, k] = model.intercept_
                #pred = model.predict(X[:, :, t])
                #accuracy[t, j, k] = accuracy_score(y, pred)

                model2 = Ridge(alpha=0.01)
                model2.fit(X[:, :, t], y)
                betas_ridge[t, (j * n_features):(j * n_features) + n_features, k] = model2.coef_
                mu_ridge[t, j, k] = model2.intercept_
                mean_error_ridge[t, j, k] = np.mean(np.abs(y - np.matmul(X[:, :, t], model2.coef_) - model2.intercept_))

                #model3 = Lasso()
                #model3.fit(X[:, :, t], y)
                #betas_lasso[t, (j * n_features):(j * n_features) + n_features, k] = model3.coef_
                #mu_lasso[t, j, k] = model3.intercept_
                #mean_error_lasso[t, j, k] = np.mean(np.abs(y - np.matmul(X[:, :, t], model3.coef_) - model3.intercept_))

    multiple_regr_dict= {#'betas':betas,
                         #'mu':mu,
                         #'accuracy':accuracy,
                         'betas_ridge': betas_ridge,
                         'mu_ridge': mu_ridge,
                         'err_ridge': mean_error_ridge}
                         #'betas_lasso': betas_lasso,
                         #'mu_lasso': mu_lasso,
                         #'err_lasso': mean_error_lasso}

    output_name = "NO_loco_dec_one_state_mutliple_regression_perm_random_states_100.pkl"

    with open(os.path.join(results_folder, output_name), 'wb') as fp:
        pickle.dump(multiple_regr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('%s - Analysis finished and results stored - %s' % (mouse_name, output_name))

# -----------------------------------------------------------------------------------------
result_dict = {}
for mouse_name in mouse_name_list:
    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name,
                                  'review_analyses')
    output_name = "NO_loco_dec_one_state_mutliple_regression_perm_random_states_100.pkl"
    with open(os.path.join(results_folder, output_name), 'rb') as fp:
        result_dict[mouse_name] = pickle.load(fp)

    n_time_points=751
    fig, ax = plt.subplots()  # (nrows=3)
    # plt.plot(np.arange(n_time_points), np.mean(multiple_regr_dict[mouse_name]['accuracy'][:,:,0],1), 'r', label='HMM accuracy rcl')
    ax.plot(np.arange(n_time_points), np.mean(result_dict[mouse_name]['err_ridge'][:,:,0], 1), 'navy', label='HMM MAE ridge')
    # plt.plot(np.arange(n_time_points), np.mean(multiple_regr_dict[mouse_name]['err_lasso'][:,:,0],1), 'k', label='HMM MAE lasso')
    # plt.plot(np.arange(n_time_points), np.mean(np.mean(multiple_regr_dict[mouse_name]['accuracy'][:,:,1:], 2),1), 'r', linestyle='-.',label='rand accuracy rcl')
    ax.plot(np.arange(n_time_points), np.mean(np.mean(result_dict[mouse_name]['err_ridge'], 2),1), 'steelblue',label='rand MAE ridge')
    # plt.plot(np.arange(n_time_points), np.mean(np.mean(multiple_regr_dict[mouse_name]['err_lasso'][:, :, 1:],2), 1), 'k',linestyle='-.',label='rand MAE lasso')
    ax.set_xticks([0, 250, 500, 750])
    # plt.legend()
    ax.set_ylim([0.1, 0.5])
    # ax2 = ax.twinx()
    # err_diff = np.mean(pval_dict[mouse_name]['err_ridge_random'], 1) - np.mean(pval_dict[mouse_name]['err_ridge_HMM'], 1)
    # ax2.bar(np.arange(751), err_diff, width=0.2, color='gray', alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('MAE')
    plt.title('average prediction error %s' % mouse_name)
    plt.show()

    pval = np.mean(result_dict[mouse_name]['err_ridge'][:, :, 0][:, :, np.newaxis] - result_dict[mouse_name]['err_ridge'][:, :, 1:]>=0, 2)
    n_time_points = 751
    fig, ax = plt.subplots()  # (nrows=3)
    # plt.plot(np.arange(n_time_points), np.mean(multiple_regr_dict[mouse_name]['accuracy'][:,:,0],1), 'r', label='HMM accuracy rcl')
    im = ax.imshow(pval.transpose(), vmin=0, vmax=0.05, cmap='inferno_r')
    fig.colorbar(im, ax=ax)

    # plt.plot(np.arange(n_time_points), np.mean(np.mean(multiple_regr_dict[mouse_name]['err_lasso'][:, :, 1:],2), 1), 'k',linestyle='-.',label='rand MAE lasso')
    ax.set_xticks([0, 250, 500, 750])

    plt.xlabel('time')
    plt.ylabel('units')
    plt.title('pvalue %s' % mouse_name)
    plt.show()



# -----------------------------------------------------------------------------------------
# retrieve results and plot

n_perms = 10000
pval_dict = {}
lags = 7
n_states = 2
mouse_name_list = ['Buchanan','Stella', 'Mitt', 'Barat']
n_time_points = 751
n_features = 3
for mouse_name in mouse_name_list:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        dirdiag = 10000000

    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,'review_analyses')
    output_name = "NO_loco_deconfounded_mutliple_regression_pval_error_10k.pkl"
    with open(os.path.join(results_folder, output_name), 'rb') as fp:
        pval_dict[mouse_name] = pickle.load(fp)

    #n_units = int((multiple_regr_dict[mouse_name]['betas_ridge'].shape[1]) / n_features)

    #unit_idx = np.arange(0, n_units * n_features, n_features)
    #unit_state_idx = np.arange(1, n_units * n_features, n_features)
    #state_idx = np.arange(2, n_units * n_features, n_features)

    #betas_cl_units = multiple_regr_dict[mouse_name]['betas'][:, unit_idx, 0]
    #betas_cl_unit_state = multiple_regr_dict[mouse_name]['betas'][:, unit_state_idx, 0]
    #betas_cl_state = multiple_regr_dict[mouse_name]['betas'][:, state_idx, 0]

    #betas_ridge_units = multiple_regr_dict[mouse_name]['betas_ridge'][:, unit_idx, 0]
    #betas_ridge_unit_state = multiple_regr_dict[mouse_name]['betas_ridge'][:, unit_state_idx, 0]
    #betas_ridge_state = multiple_regr_dict[mouse_name]['betas_ridge'][:, state_idx, 0]

    #betas_lasso_units = multiple_regr_dict[mouse_name]['betas_lasso'][:, unit_idx, 0]
    #betas_lasso_unit_state = multiple_regr_dict[mouse_name]['betas_lasso'][:, unit_state_idx, 0]
    #betas_lasso_state = multiple_regr_dict[mouse_name]['betas_lasso'][:, state_idx, 0]

    #pvalue_acc_cl = np.sum(multiple_regr_dict[mouse_name]['accuracy'][:,:,0][:,:,np.newaxis]<=multiple_regr_dict[mouse_name]['accuracy'][:,:,1:], axis=2)/ n_perms
    #pvalue_ridge = np.sum(multiple_regr_dict[mouse_name]['err_ridge'][:,:,0][:,:,np.newaxis]>=multiple_regr_dict[mouse_name]['err_ridge'][:,:,1:], axis=2)/ n_perms
    #pvalue_lasso = np.sum(multiple_regr_dict[mouse_name]['err_lasso'][:,:,0][:,:,np.newaxis]>=multiple_regr_dict[mouse_name]['err_lasso'][:,:,1:], axis=2)/ n_perms

    # plot errors and accuracy
    fig, ax = plt.subplots()  # (nrows=3)
    #plt.plot(np.arange(n_time_points), np.mean(multiple_regr_dict[mouse_name]['accuracy'][:,:,0],1), 'r', label='HMM accuracy rcl')
    ax.plot(np.arange(n_time_points), np.mean(pval_dict[mouse_name]['err_ridge_HMM'],1), 'navy', label='HMM MAE ridge')
    #plt.plot(np.arange(n_time_points), np.mean(multiple_regr_dict[mouse_name]['err_lasso'][:,:,0],1), 'k', label='HMM MAE lasso')
    #plt.plot(np.arange(n_time_points), np.mean(np.mean(multiple_regr_dict[mouse_name]['accuracy'][:,:,1:], 2),1), 'r', linestyle='-.',label='rand accuracy rcl')
    ax.plot(np.arange(n_time_points), np.mean(pval_dict[mouse_name]['err_ridge_random'], 1), 'steelblue',label='rand MAE ridge')
    #plt.plot(np.arange(n_time_points), np.mean(np.mean(multiple_regr_dict[mouse_name]['err_lasso'][:, :, 1:],2), 1), 'k',linestyle='-.',label='rand MAE lasso')
    ax.set_xticks([0, 250, 500, 750])
    #plt.legend()
    ax.set_ylim([0.1,0.5])
    ax2 = ax.twinx()
    err_diff = np.mean(pval_dict[mouse_name]['err_ridge_random'], 1)- np.mean(pval_dict[mouse_name]['err_ridge_HMM'],1)
    ax2.bar(np.arange(751),err_diff,width=0.2, color='gray',alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('MAE')
    plt.title('average prediction error %s' % mouse_name)
    plt.show()

    # -----------------------------------------------------------------------------------------------
    # --------------------- plot the cluster-based corrected pvalue --------------------------------
    # create a pvalue matrix
    # clusters[0] = 0,0
    # clusters[1] = 0,1
    # clusters[2] = 0,2

    #img_cluster_based_p = pval_dict[mouse_name]['cluster_pvalue'].reshape(pval_dict[mouse_name]['pval_uncorrected'].shape)
    fig, ax = plt.subplots()  # (nrows=3)

    im2 = ax.imshow(pval_dict[mouse_name]['pval_uncorrected'], cmap='inferno_r', vmin=0, vmax=0.05)
    # ax[1].set_title('error ridge ')
    # im3= ax[2].imshow(pvalue_lasso.transpose(1,0), cmap='plasma', vmin=0, vmax=0.05)
    # ax[2].set_title('error lasso ')
    # ax.set_xticks([0, 250, 500, 750])
    # ax.set_xticklabels(['-1', '0', '1', '2'])
    fig.colorbar(im2, ax=ax, location='bottom', orientation='horizontal')
    # fig.colorbar(im2, ax=[ax[0],ax[1],ax[2]], location='right', orientation='vertical')
    fig.suptitle('pvalues, uncorrected %s' % mouse_name)
    plt.show()

    from glhmm import statistics
    corrected_bhy, significant_bhy = statistics.pval_correction({'pval': pval_dict[mouse_name]['pval_betas_ridge'].transpose(1,0)}, method='fdr_by', alpha=0.01)
    corrected_bon, significant_bon = statistics.pval_correction(
        {'pval': pval_dict[mouse_name]['pval_betas_ridge'].transpose(1, 0)}, method='bonferroni', alpha=0.01)
    # plot pvalue
    fig, ax = plt.subplots()#(nrows=3)
    #im1 = ax[0].imshow(pvalue_acc_cl.transpose(1,0), cmap='plasma', vmin=0, vmax=0.05)
    #ax[0].set_title('accuracy ridge classifier')
    #ax[0].vlines(251, 0, n_units, 'k', linestyles='dashdot')
    #im2 = ax.imshow(pval_dict[mouse_name]['pval_betas_ridge'].transpose(1,0)*2, cmap='inferno_r', vmin=0, vmax=0.05)
    im2 = ax.imshow(corrected_bhy, cmap='inferno_r', vmin=0, vmax=0.05)
    #ax[1].set_title('error ridge ')
    #im3= ax[2].imshow(pvalue_lasso.transpose(1,0), cmap='plasma', vmin=0, vmax=0.05)
    #ax[2].set_title('error lasso ')
    # ax.set_xticks([0, 250, 500, 750])
    # ax.set_xticklabels(['-1', '0', '1', '2'])
    fig.colorbar(im2, ax=ax,location='bottom', orientation='horizontal')
    #fig.colorbar(im2, ax=[ax[0],ax[1],ax[2]], location='right', orientation='vertical')
    fig.suptitle('pvalues, corrected b-y %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots(nrows=2)
    im1 = ax[0].imshow(betas_ridge_units.transpose(1, 0), cmap='coolwarm', vmin=-2, vmax=2)
    ax[0].set_title('state 1')
    # ax[0].vlines(251, 0, n_units, 'k', linestyles='dashdot')
    im2 = ax[1].imshow(betas_ridge_unit_state.transpose(1, 0), cmap='coolwarm', vmin=-2, vmax=2)
    ax[1].set_title('states difference')
    #im3 = ax[2].imshow(betas_ridge_units.transpose(1, 0) + betas_ridge_unit_state.transpose(1, 0), cmap='coolwarm', vmin=-2, vmax=2)
    #ax[2].set_title('state 2')
    # ax.set_xticks([0, 250, 500, 750])
    # ax.set_xticklabels(['-1', '0', '1', '2'])
    # fig.colorbar(im1, ax=[ax[0]],location='bottom', orientation='horizontal')
    fig.colorbar(im2, ax=[ax[0], ax[1]], location='right', orientation='vertical')
    fig.suptitle('betas, %s' % mouse_name)
    plt.show()

    # plot pvalue
    fig, ax = plt.subplots()
    im2 = ax.imshow(nonzero_beta_mask[mouse_name].transpose(1, 0) * betas_ridge_unit_state.transpose(1, 0), cmap='coolwarm', vmin=-2, vmax=2)
    fig.colorbar(im2, ax=ax, location='bottom', orientation='horizontal')
    fig.suptitle('nonzero betas, %s' % mouse_name)
    plt.show()

    """
    fig, ax = plt.subplots(nrows=3)
    im1 = ax[0].imshow(betas_cl_unit_state.transpose(1,0), cmap='coolwarm', vmin=-1, vmax=1)
    ax[0].set_title('ridge classifier')
    # ax[0].vlines(251, 0, n_units, 'k', linestyles='dashdot')
    im2 = ax[1].imshow(betas_ridge_unit_state.transpose(1,0), cmap='coolwarm', vmin=-1, vmax=1)
    ax[1].set_title('ridge ')
    im3 = ax[2].imshow(betas_lasso_unit_state.transpose(1,0), cmap='coolwarm', vmin=-1, vmax=1)
    ax[2].set_title('lasso ')
    # ax.set_xticks([0, 250, 500, 750])
    # ax.set_xticklabels(['-1', '0', '1', '2'])
    # fig.colorbar(im1, ax=[ax[0]],location='bottom', orientation='horizontal')
    fig.colorbar(im2, ax=[ax[0], ax[1], ax[2]], location='right', orientation='vertical')
    fig.suptitle('betas, %s' % mouse_name)
    plt.show()
    """

# ------------------- use only one state for the regression -----------------------
# ------------------- bootstrap for confidence intervals -----------------------


n_bootstrap = 100

mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']

for mouse_name in mouse_name_list:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        dirdiag = 10000000
        lags = 7
        n_states = 2

    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        dirdiag = 1000000000
        lags = 7
        n_states = 2

    # if on local
    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
    info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                                    mouse_name)

    # gammas_response_name = 'pca_%s_gammas_aligned_pokeout_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)
    gammas_response_name = 'pca_%s_vp_aligned_pokeout_orig_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)

    info_file = '%s_trial_info.npy' % mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))

    success_labels = info_data[:, 0]
    inseq_labels = info_data[:, 1]
    odor_labels = info_data[:, 3]

    units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s' % mouse_name
    units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

    with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
        units_density_response = pickle.load(fp)

    gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
    # if with prob
    # states_around_response = gammas_trials_response['states_around_pokeout'].transpose(0,2,1)
    # if with vp
    states_around_response = gammas_trials_response['states_around_pokeout_orig']
    gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)

    # -------------------------------- retrieve useful info -------------------------------------
    n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
    # choose kernel 1
    dk = 1
    n_time_points = 751

    # if with prob
    # gamma = states_around_response[:, 0, 250:]
    gamma = gamma_vp_response[:, 0, 250:] * 2.0 - 1.0
    y = success_labels * 2.0 - 1.0

    X_units = units_density_response[:, :, 250:, dk]
    fakegammas = np.zeros(shape=(n_trials, 1, n_time_points))
    # either normalize the units this way or per state
    # X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

    # normalize the units per state
    # run it with standardised units activity per state
    for t in range(n_time_points):
        y_t = gamma[:, t]
        X_t = X_units[:, :, t]

        X1 = X_t[y_t == -1, :]
        X1_normalized = (X1 - np.mean(X1, 1)[:, np.newaxis]) / np.std(X1, 1)[:, np.newaxis]
        X2 = X_t[y_t == 1, :]
        X2_normalized = (X2 - np.mean(X2, 1)[:, np.newaxis]) / np.std(X2, 1)[:, np.newaxis]

        X_t[y_t == -1, :] = X1_normalized
        X_t[y_t == 1, :] = X2_normalized

        X_units[:, :, t] = X_t

    # first compute it with no perms, take it as first to compare
    # at this point I should have my X and my Y
    # start with regression with bootstrap at each timepoint
    n_features = 3
    # betas = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
    # mu = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    betas_ridge = np.empty(shape=(n_time_points, n_units * n_features, n_bootstrap + 1))
    mu_ridge = np.empty(shape=(n_time_points, n_units, n_bootstrap + 1))
    # betas_lasso = np.empty(shape=(n_time_points, n_units * n_features, n_perms + 1))
    # mu_lasso = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    mean_error_ridge = np.empty(shape=(n_time_points, n_units, n_bootstrap + 1))
    # accuracy = np.empty(shape=(n_time_points, n_units, n_perms + 1))
    # mean_error_lasso = np.empty(shape=(n_time_points, n_units, n_perms + 1))

    units_mult_gamma1 = X_units * gamma[:, np.newaxis, :]


    # the first one is the basic statistic: run the regression per neuron, no swapping
    for j in range(n_units):
        # took away gamma because unable to allocate that much memory
        X = np.concatenate(
            (X_units[:, j, :][:, np.newaxis, :], units_mult_gamma1[:, j, :][:, np.newaxis, :], fakegammas),
            axis=1)
        # first run the unpermuted regression
        for t in range(n_time_points):
            # first the original, unshuffled
            # model = RidgeClassifier(alpha=0.001)
            # model.fit(X[:, :, t], y)
            # betas[t, (j * n_features):(j * n_features) + n_features, 0] = model.coef_
            # mu[t, j, 0] = model.intercept_
            # pred = model.predict(X[:,:,t])
            # accuracy[t,j,0] = accuracy_score(y,pred)

            model2 = Ridge(alpha=0.001)
            model2.fit(X[:, :, t], y)
            betas_ridge[t, (j * n_features):(j * n_features) + n_features, 0] = model2.coef_
            mu_ridge[t, j, 0] = model2.intercept_
            mean_error_ridge[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model2.coef_) - model2.intercept_))

            # model3 = Lasso()
            # model3.fit(X[:, :, t], y)
            # betas_lasso[t, (j * n_features):(j * n_features) + n_features, 0] = model3.coef_
            # mu_lasso[t, j, 0] = model3.intercept_
            # mean_error_lasso[t, j, 0] = np.mean(np.abs(y - np.matmul(X[:, :, t], model3.coef_) - model3.intercept_))

    # then we do with the permutations swapping gammas, the same index for all neurons and for all time points.
    # swap for each iteration

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
                                fakegammas), axis=1)
            for t in range(n_time_points):
                # model = RidgeClassifier(alpha=0.001)
                # model.fit(X[:, :, t], y)
                # betas[t, (j * n_features):(j * n_features) + n_features, k] = model.coef_
                # mu[t, j, k] = model.intercept_
                # pred = model.predict(X[:, :, t])
                # accuracy[t, j, k] = accuracy_score(y, pred)

                model2 = Ridge(alpha=0.001)
                model2.fit(X_boot[:, :, t], y_boot)
                betas_ridge[t, (j * n_features):(j * n_features) + n_features, k] = model2.coef_
                mu_ridge[t, j, k] = model2.intercept_
                mean_error_ridge[t, j, k] = np.mean(np.abs(y_boot - np.matmul(X_boot[:, :, t], model2.coef_) - model2.intercept_))

                # model3 = Lasso()
                # model3.fit(X[:, :, t], y)
                # betas_lasso[t, (j * n_features):(j * n_features) + n_features, k] = model3.coef_
                # mu_lasso[t, j, k] = model3.intercept_
                # mean_error_lasso[t, j, k] = np.mean(np.abs(y - np.matmul(X[:, :, t], model3.coef_) - model3.intercept_))

    multiple_regr_dict = {  # 'betas':betas,
        # 'mu':mu,
        # 'accuracy':accuracy,
        'betas_ridge': betas_ridge,
        'mu_ridge': mu_ridge,
        'err_ridge': mean_error_ridge}
    # 'betas_lasso': betas_lasso,
    # 'mu_lasso': mu_lasso,
    # 'err_lasso': mean_error_lasso}

    output_name = "vp_one_state_mutliple_regression_state_centered_stand_per_state_bootstrap_100.pkl"

    with open(os.path.join(results_folder, output_name), 'wb') as fp:
        pickle.dump(multiple_regr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('%s - Analysis finished and results stored - %s' % (mouse_name, output_name))



# -----------------------------------------------------------------------------------------
# retrieve results and plot

n_bootstrap = 100
multiple_regr_dict = {}
lags = 7
n_states = 2
mouse_name_list = ['Buchanan','Stella', 'Mitt', 'Barat']
n_time_points = 751
n_features = 3
nonzero_beta_mask = {}
for mouse_name in mouse_name_list:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        dirdiag = 10000000

    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,'review_analyses')
    output_name = "vp_one_state_mutliple_regression_state_centered_stand_per_state_bootstrap_100.pkl"
    with open(os.path.join(results_folder, output_name), 'rb') as fp:
        multiple_regr_dict[mouse_name] = pickle.load(fp)

    n_units = int((multiple_regr_dict[mouse_name]['betas_ridge'].shape[1]) / n_features)

    unit_idx = np.arange(0, n_units * n_features, n_features)
    unit_state_idx = np.arange(1, n_units * n_features, n_features)
    state_idx = np.arange(2, n_units * n_features, n_features)

    # now we want to check that this unit-state weight is non zero
    #betas_ridge_units = multiple_regr_dict[mouse_name]['betas_ridge'][:, unit_idx, 0]
    betas_ridge_unit_state = multiple_regr_dict[mouse_name]['betas_ridge'][:, unit_state_idx,:]
    #betas_ridge_state = multiple_regr_dict[mouse_name]['betas_ridge'][:, state_idx, 0]

    # order the betas
    betas_ordered = np.sort(betas_ridge_unit_state, axis=2)
    betas_lb = np.sign(betas_ordered[:,:,5])
    betas_ub = np.sign(betas_ordered[:,:,95])
    betas_nonzero = (betas_lb*betas_ub)>0

    nonzero_beta_mask[mouse_name] = betas_nonzero

    # plot pvalue
    fig, ax = plt.subplots()
    im2 = ax.imshow(betas_nonzero.transpose(1, 0), cmap='plasma', vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax, location='bottom', orientation='horizontal')
    fig.suptitle('nonzero betas, %s' % mouse_name)
    plt.show()




