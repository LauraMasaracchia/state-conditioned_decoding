
# data exploration of Fortin Lab data

from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LinearRegression, RidgeClassifier
from glhmm import statistics
from Decoding_analysis.Fortin_data_analysis.utils.functions import *
from Decoding_analysis.Fortin_data_analysis.utils.preproc import *
from itertools import permutations, combinations

# ----------------------- perform analyses without movement trials ------------------
# -----------------------------------------------------------------------------
# ----------------- first trials without movement, -----------------------------------
# -------- then take out successful trials with movement
# --------- meaning that the movement difference between success and failure is systematic -----------------------------
# HMM gammas specifications
# gammas at response have -500 and +500 around response
# gammas at stimulus onset are -250 to +750 around stimulus onset
superchris_bad_trials = [0, 1, 4, 10, 137]

# modulations of
data_aligned_strings = 'response'
decoding_strings = ['success']  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap = 100
min_trial_nbr = 4


for freq_band in ['orig']:
    print('Analyses start in freq band %s'%freq_band)
    result_rat_dict = {}
    mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']
    bs_stored_prediction = {}
    bs_stored_prediction_units = {}
    bs_stored_true_label_units = {}
    # add permutations of randomly selected trials (while still keeping the class balanc

    for mouse_name in mouse_name_list:
        print(mouse_name)
        n_states = 2
        lags = 7
        if mouse_name == 'Mitt':
            dirdiag = 10000000
        else:
            dirdiag = 1000000000

        results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
        gammas_response_name = 'pca_%s_vp_aligned_pokeout_%s_lag%d_k%d_dirdiag%d.mat' % (mouse_name,freq_band, lags, n_states, dirdiag)

        info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                                        mouse_name)
        info_file = '%s_trial_info.npy' % mouse_name.lower()
        info_data = np.load(os.path.join(info_data_folder, info_file))

        # load movement trials data
        movement_trials = scipy.io.loadmat(os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,'review_analyses','trials_to_discard.mat'))
        movement_trials_response = movement_trials['trials_locomotion'][0]-1

        info_data = np.delete(info_data, movement_trials_response, 0)

        success_labels = info_data[:, 0]
        inseq_labels = info_data[:, 1]
        odor_labels = info_data[:, 3]

        units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s' % mouse_name
        units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

        with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
            units_density_response = pickle.load(fp)

        # delete the trials also from gammas and unit data
        gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
        states_around_response = gammas_trials_response['states_around_pokeout_%s'%freq_band]

        # -------------------------------- retrieve useful info -------------------------------------
        units_density_response = np.delete(units_density_response, movement_trials_response, 0)
        n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
        # choose kernel 1
        dk = 1
        print(n_trials)

        # ----------------------------- get the gammas shape from vpath -----------------------------
        # convert vp to gammas shape
        gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
        # check the dimensions are correct
        gamma_vp_response = np.delete(gamma_vp_response, movement_trials_response, 0)
        n_states = gamma_vp_response.shape[1]

        # --------------------------------------------------------------------------------------------------------
        # -------------------------- analyses on gammas and units ---------------------------

        X = units_density_response[:, :, :, dk]
        X = (X - np.mean(X, 1)[:, np.newaxis,:]) / np.std(X, 1)[:, np.newaxis,:]
        gamma = gamma_vp_response

        for decode in decoding_strings:
            print('decoding %s' % decode)
            if decode == 'success':
                y = success_labels
                y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

            elif decode == 'trial_type':
                y = inseq_labels
                y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

            acc_standard = np.zeros(shape=(n_time_points, n_bootstrap))
            acc_per_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))
            acc_standard_per_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))
            acc_cross_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))

            estimated_betas_states = np.zeros(shape=(n_states, n_units, n_time_points, n_bootstrap))
            estimated_betas_standard = np.zeros(shape=(n_units, n_time_points, n_bootstrap))
            intercept_standard = np.zeros(shape=(n_time_points, n_bootstrap))
            intercept_states = np.zeros(shape=(n_states, n_time_points, n_bootstrap))

            acc_per_state_general = np.empty(shape=(n_time_points,))
            acc_per_state_general[:] = np.nan
            acc_cross_state_general = np.empty(shape=(n_time_points,))
            acc_cross_state_general[:] = np.nan
            acc_standard_general = np.empty(shape=(n_time_points,))
            acc_standard_general[:] = np.nan
            acc_tot_avg = np.zeros(shape=(n_time_points, n_bootstrap))

            train_trials = np.zeros(shape=(n_time_points, 4))

            # save predictions
            y_pred_standard_all = {}
            y_true_standard_all = {}
            y_pred_standard_state0 = {}
            y_pred_standard_state1 = {}
            y_true_state0 = {}
            y_pred_state0 = {}
            y_true_state1 = {}
            y_pred_state1 = {}
            y_pred_cross_state01 = {}
            y_pred_cross_state10 = {}

            for t in range(n_time_points):
                print('time point t = %d' % t)
                # for each time point
                # group trials by behavior or by state
                sorted_trials = sort_trials_by_variables(y[:, t], gamma[:, 1, t])
                train_trials[t, :] = len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]), len(
                    sorted_trials[1][1])
                # spitted as:
                # 0A , 0B
                # 1A , 1B

                y_pred_standard_all[t] = []
                y_true_standard_all[t] = []
                y_pred_standard_state0[t] = []
                y_pred_standard_state1[t] = []
                y_true_state0[t] = []
                y_pred_state0[t] = []
                y_true_state1[t] = []
                y_pred_state1[t] = []
                y_pred_cross_state01[t] = []
                y_pred_cross_state10[t] = []
                # for each fold
                if min(len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]),
                       len(sorted_trials[1][1])) < 5:
                    continue

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

                    # ------------------------------------- STATE 1 -----------------------------------------
                    # ------------------------- create train and test data for this fold -----------------------
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

                    # ------------------------------------------------------------------------------------------------------
                    # ------------------ training phase : train a model for all and one for each state ---------------------

                    tot_train_trials = np.concatenate((tot_train_trials_state0,tot_train_trials_state1),0)
                    tot_test_trials = np.concatenate((tot_test_trials_state0, tot_test_trials_state1), 0)
                    np.random.shuffle(tot_train_trials)

                    model_standard = RidgeClassifierCV(alphas=np.array([0.01,0.05,0.1,0.5,1.0,5,10,50,100,500]))
                    model_state0 = RidgeClassifierCV(alphas=np.array([0.01,0.1,1,10,50,100,500]))
                    model_state1 = RidgeClassifierCV(alphas=np.array([0.01,0.1,1,10,50,100,500]))
                    model_avg = RidgeClassifier()

                    # train standard model
                    model_standard.fit(X[tot_train_trials,:,t],y[tot_train_trials,t])
                    model_avg.fit(X[tot_train_trials,:,t],y[tot_train_trials,t])
                    estimated_betas_standard[:,t,f] = model_standard.coef_
                    #intercept_standard[t,f] = model_standard.intercept_

                    # test standard model on all
                    y_standard_pred = model_standard.predict(X[tot_test_trials,:,t])

                    # accuracy here
                    acc_standard[t, f] = accuracy_score(y[tot_test_trials, t], y_standard_pred)

                    # test standard model on each state
                    y_standard_pred_state0 = model_standard.predict(X[tot_test_trials_state0, :, t])
                    acc_standard_per_state[0,t,f] = accuracy_score(y[tot_test_trials_state0,t],y_standard_pred_state0)
                    y_standard_pred_state1 = model_standard.predict(X[tot_test_trials_state1, :, t])
                    acc_standard_per_state[1, t, f] = accuracy_score(y[tot_test_trials_state1, t], y_standard_pred_state1)

                    # train per state models
                    model_state0.fit(X[tot_train_trials_state0, :, t], y[tot_train_trials_state0, t])
                    estimated_betas_states[0,:, t, f] = model_state0.coef_
                    #intercept_states[0,t,f] = model_state0.intercept_
                    model_state1.fit(X[tot_train_trials_state1, :, t], y[tot_train_trials_state1, t])
                    estimated_betas_states[1, :, t, f] = model_state1.coef_
                    #intercept_states[1, t, f] = model_state1.intercept_

                    model_avg.coef_ = (model_state1.coef_ + model_state0.coef_) / 2
                    #model_avg.intercept_ = (model_state1.intercept_ + model_state0.intercept_) / 2
                    y_avg_pred = model_avg.predict(X[tot_test_trials, :, t])
                    acc_tot_avg[t, f] = accuracy_score(y[tot_test_trials, t], y_avg_pred)
                    # test model states within a state
                    y_state0_pred_state0 = model_state0.predict(X[tot_test_trials_state0,:,t])
                    acc_per_state[0,t,f] = accuracy_score(y[tot_test_trials_state0,t],y_state0_pred_state0)
                    y_state1_pred_state1 = model_state1.predict(X[tot_test_trials_state1,:,t])
                    acc_per_state[1, t, f] = accuracy_score(y[tot_test_trials_state1, t], y_state1_pred_state1)

                    # test state models across state
                    y_state0_pred_state1 = model_state0.predict(X[tot_test_trials_state1,:,t])
                    acc_cross_state[1,t,f] = accuracy_score(y[tot_test_trials_state1, t],y_state0_pred_state1)
                    y_state1_pred_state0 = model_state1.predict(X[tot_test_trials_state0,:,t])
                    acc_cross_state[0, t, f] = accuracy_score(y[tot_test_trials_state0, t], y_state1_pred_state0)

                    # save all predictions
                    y_pred_standard_all[t].extend(y_standard_pred)
                    y_true_standard_all[t].extend(y[tot_test_trials,t])
                    y_pred_standard_state0[t].extend(y_standard_pred_state0)
                    y_pred_standard_state1[t].extend(y_standard_pred_state1)
                    y_true_state0[t].extend(y[tot_test_trials_state0,t])
                    y_pred_state0[t].extend(y_state0_pred_state0)
                    y_true_state1[t].extend(y[tot_test_trials_state1,t])
                    y_pred_state1[t].extend(y_state1_pred_state1)
                    y_pred_cross_state01[t].extend(y_state0_pred_state1)
                    y_pred_cross_state10[t].extend(y_state1_pred_state0)

                acc_standard_general[t] = accuracy_score(y_true_standard_all[t],y_pred_standard_all[t])
                y_true_state_all = np.concatenate((y_true_state0[t],y_true_state1[t]),0)
                y_pred_state_all = np.concatenate((y_pred_state0[t],y_pred_state1[t]),0)
                acc_per_state_general[t] = accuracy_score(y_true_state_all,y_pred_state_all)
                y_cross_state_all = np.concatenate((y_pred_cross_state10[t], y_pred_cross_state01[t]), 0)

                acc_cross_state_general[t] = accuracy_score(y_true_state_all,y_cross_state_all)

                #acc_standard_general[t] = accuracy_score()


        output_name = "no_mov_standardised_%s_units_per_state_decoding_%s-nbootstrap_%d_state.pkl" % (model_name, decode, n_bootstrap)

        result_rat_dict[mouse_name] = {"acc_standard": acc_standard,'acc_standard_per_state':acc_standard_per_state,
                                       'acc_per_state': acc_per_state,'acc_cross_state': acc_cross_state,
                                       'acc_tot_avg': acc_tot_avg,
                                       'acc_standard_general': acc_standard_general,
                                       'acc_per_state_general': acc_per_state_general,
                                       'acc_cross_state_general':acc_cross_state_general,
                                       'estimated_betas_standard':estimated_betas_standard,
                                       'estimated_betas_states': estimated_betas_states,
                                       'intercept_standard': intercept_standard,
                                       'intercept_states': intercept_states}

        with open(os.path.join(results_folder, output_name), 'wb') as fp:
            pickle.dump(result_rat_dict[mouse_name], fp, protocol=pickle.HIGHEST_PROTOCOL)

        print('%s - Analysis finished and results stored - %s' % (mouse_name, freq_band))


        mean_acc_standard, interval_acc_standard = compute_CI(result_rat_dict[mouse_name]['acc_standard'], axis=1)
        mean_acc_standard_state0, interval_acc_standard_state0 = compute_CI(result_rat_dict[mouse_name]['acc_standard_per_state'][0,:,:], axis=1)
        mean_acc_standard_state1, interval_acc_standard_state1 = compute_CI(result_rat_dict[mouse_name]['acc_standard_per_state'][1,:,:], axis=1)
        mean_acc_state0, interval_acc_state0 = compute_CI(result_rat_dict[mouse_name]['acc_per_state'][0, :, :], axis=1)
        mean_acc_state1, interval_acc_state1 = compute_CI(result_rat_dict[mouse_name]['acc_per_state'][1, :, :], axis=1)
        mean_acc_cross_state01, interval_acc_cross_state01 = compute_CI(result_rat_dict[mouse_name]['acc_cross_state'][1, :, :], axis=1)
        mean_acc_cross_state10, interval_acc_cross_state10 = compute_CI(result_rat_dict[mouse_name]['acc_cross_state'][0, :, :], axis=1)
        mean_acc_tot_avg, interval_acc_tot_avg = compute_CI(result_rat_dict[mouse_name]['acc_tot_avg'], axis=1)

        # plot decoding results with CI
        fig, ax = plt.subplots()
        time_scale = np.arange(0, n_time_points)
        ax.plot(time_scale, mean_acc_standard, color='navy', label='standard')
        ax.fill_between(time_scale, interval_acc_standard[0], interval_acc_standard[1], alpha=0.5,color='navy')
        #ax.plot(time_scale, mean_acc_tot_avg, color='steelblue', label='avg_tot')
        #ax.fill_between(time_scale, interval_acc_tot_avg[0], interval_acc_tot_avg[1], alpha=0.5, color='steelblue')
        ax.plot(time_scale, mean_acc_state0, color='crimson', label='y1,beta1')
        ax.fill_between(time_scale, interval_acc_state0[0], interval_acc_state0[1], alpha=0.5,color='crimson')
        ax.plot(time_scale, mean_acc_state1, color='orange', label='y2,beta2')
        ax.fill_between(time_scale, interval_acc_state1[0], interval_acc_state1[1], alpha=0.5, color='orange')

        #ax.plot(time_scale, mean_acc_standard_state0, color='steelblue', label='y1,beta0')
        #ax.fill_between(time_scale, interval_acc_standard_state0[0], interval_acc_standard_state0[1], alpha=0.5,color='steelblue')
        #ax.plot(time_scale, mean_acc_standard_state1, color='green', label='y2,beta0')
        #ax.fill_between(time_scale, interval_acc_standard_state1[0], interval_acc_standard_state1[1], alpha=0.5,color='green')

        #ax.plot(time_scale, mean_acc_cross_state01, color='pink', label='y2,beta1')
        #ax.fill_between(time_scale, interval_acc_cross_state01[0], interval_acc_cross_state01[1], alpha=0.5, color='pink')
        #ax.plot(time_scale, mean_acc_cross_state10, color='violet', label='y1,beta2')
        #ax.fill_between(time_scale, interval_acc_cross_state10[0], interval_acc_cross_state10[1], alpha=0.5, color='violet')

        ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
        ax.vlines(501, 0, 1, 'k', linestyles='dashdot')
        plt.legend()
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
        ax.set_xlabel('t [s]')
        ax.set_xlim([250, 1001])
        ax.set_ylim([0.25, 1.05])
        ax.set_ylabel('accuracy')
        plt.title('NO LOCOMOTION, per state decoding success, %s' % mouse_name)
        plt.show()


        # plot aggregated decoding results
        fig, ax = plt.subplots()
        time_scale = np.arange(0, n_time_points)
        ax.plot(time_scale,result_rat_dict[mouse_name]['acc_standard_general'], color='navy', label='standard')
        ax.plot(time_scale, result_rat_dict[mouse_name]['acc_per_state_general'], color='orangered', label='per-state')
        ax.plot(time_scale, result_rat_dict[mouse_name]['acc_cross_state_general'], color='purple', label='cross-state')
        ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
        ax.vlines(501, 0, 1, 'k', linestyles='dashdot')
        plt.legend()
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
        ax.set_xlabel('t [s]')
        ax.set_xlim([250, 1001])
        ax.set_ylim([0.25, 1.05])
        ax.set_ylabel('accuracy')
        plt.title('NO LOCOMOTION, Aggregated measures, decoding success, %s' % mouse_name)
        plt.show()


# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------- decoding only with movement --------------------------------------------
# get those trials where there is movement (every failed trial and every successful last sequence trial)
# and do the state-conditioned decoding with those

# modulations of
data_aligned_strings = 'response'
decoding_strings = ['success']  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap = 200
min_trial_nbr = 4


for freq_band in ['orig']:
    print('Analyses start in freq band %s'%freq_band)
    result_rat_dict = {}
    mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']
    bs_stored_prediction = {}
    bs_stored_prediction_units = {}
    bs_stored_true_label_units = {}
    # add permutations of randomly selected trials (while still keeping the class balanc

    for mouse_name in mouse_name_list:
        print(mouse_name)
        n_states = 2
        lags = 7
        if mouse_name == 'Mitt':
            dirdiag = 10000000
        else:
            dirdiag = 1000000000

        results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
        gammas_response_name = 'pca_%s_vp_aligned_pokeout_%s_lag%d_k%d_dirdiag%d.mat' % (mouse_name,freq_band, lags, n_states, dirdiag)

        info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                                        mouse_name)
        info_file = '%s_trial_info.npy' % mouse_name.lower()
        info_data = np.load(os.path.join(info_data_folder, info_file))

        # load movement trials data
        movement_trials = scipy.io.loadmat(os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,'review_analyses','trials_to_discard.mat'))
        movement_trials_response = movement_trials['trials_with_supposed_motion'].squeeze()-1

        invalid_trials = list(set(np.arange(len(info_data))) - set(movement_trials_response))
        invalid_trials = np.array(invalid_trials)

        info_data = np.delete(info_data, invalid_trials, 0)

        success_labels = info_data[:, 0]
        inseq_labels = info_data[:, 1]
        odor_labels = info_data[:, 3]

        units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s' % mouse_name
        units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

        with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
            units_density_response = pickle.load(fp)

        # delete the trials also from gammas and unit data
        gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
        states_around_response = gammas_trials_response['states_around_pokeout_%s'%freq_band]

        # -------------------------------- retrieve useful info -------------------------------------
        units_density_response = np.delete(units_density_response, invalid_trials, 0)
        n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
        # choose kernel 1
        dk = 1

        # ----------------------------- get the gammas shape from vpath -----------------------------
        # convert vp to gammas shape
        gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)
        # check the dimensions are correct
        gamma_vp_response = np.delete(gamma_vp_response, invalid_trials, 0)
        n_states = gamma_vp_response.shape[1]

        # --------------------------------------------------------------------------------------------------------
        # -------------------------- analyses on gammas and units ---------------------------

        X = units_density_response[:, :, :, dk]
        X = (X - np.mean(X, 1)[:, np.newaxis,:]) / np.std(X, 1)[:, np.newaxis,:]
        gamma = gamma_vp_response

        for decode in decoding_strings:
            print('decoding %s' % decode)
            if decode == 'success':
                y = success_labels
                y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

            elif decode == 'trial_type':
                y = inseq_labels
                y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

            acc_standard = np.zeros(shape=(n_time_points, n_bootstrap))
            acc_per_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))
            acc_standard_per_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))
            acc_cross_state = np.zeros(shape=(n_states, n_time_points, n_bootstrap))

            estimated_betas_states = np.zeros(shape=(n_states, n_units, n_time_points, n_bootstrap))
            estimated_betas_standard = np.zeros(shape=(n_units, n_time_points, n_bootstrap))
            intercept_standard = np.zeros(shape=(n_time_points, n_bootstrap))
            intercept_states = np.zeros(shape=(n_states, n_time_points, n_bootstrap))

            acc_per_state_general = np.empty(shape=(n_time_points,))
            acc_per_state_general[:] = np.nan
            acc_cross_state_general = np.empty(shape=(n_time_points,))
            acc_cross_state_general[:] = np.nan
            acc_standard_general = np.empty(shape=(n_time_points,))
            acc_standard_general[:] = np.nan
            acc_tot_avg = np.zeros(shape=(n_time_points, n_bootstrap))

            train_trials = np.zeros(shape=(n_time_points, 4))

            # save predictions
            y_pred_standard_all = {}
            y_true_standard_all = {}
            y_pred_standard_state0 = {}
            y_pred_standard_state1 = {}
            y_true_state0 = {}
            y_pred_state0 = {}
            y_true_state1 = {}
            y_pred_state1 = {}
            y_pred_cross_state01 = {}
            y_pred_cross_state10 = {}

            for t in range(n_time_points):
                print('time point t = %d' % t)
                # for each time point
                # group trials by behavior or by state
                sorted_trials = sort_trials_by_variables(y[:, t], gamma[:, 1, t])
                train_trials[t, :] = len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]), len(
                    sorted_trials[1][1])
                # spitted as:
                # 0A , 0B
                # 1A , 1B

                y_pred_standard_all[t] = []
                y_true_standard_all[t] = []
                y_pred_standard_state0[t] = []
                y_pred_standard_state1[t] = []
                y_true_state0[t] = []
                y_pred_state0[t] = []
                y_true_state1[t] = []
                y_pred_state1[t] = []
                y_pred_cross_state01[t] = []
                y_pred_cross_state10[t] = []
                # for each fold
                if min(len(sorted_trials[0][0]), len(sorted_trials[1][0]), len(sorted_trials[0][1]),
                       len(sorted_trials[1][1])) < 5:
                    continue

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

                    # ------------------------------------- STATE 1 -----------------------------------------
                    # ------------------------- create train and test data for this fold -----------------------
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

                    # ------------------------------------------------------------------------------------------------------
                    # ------------------ training phase : train a model for all and one for each state ---------------------

                    tot_train_trials = np.concatenate((tot_train_trials_state0,tot_train_trials_state1),0)
                    tot_test_trials = np.concatenate((tot_test_trials_state0, tot_test_trials_state1), 0)
                    np.random.shuffle(tot_train_trials)

                    model_standard = RidgeClassifierCV(alphas=np.array([0.01,0.05,0.1,0.5,1.0,5,10,50,100,500]))
                    model_state0 = RidgeClassifierCV(alphas=np.array([0.01,0.1,1,10,50,100,500]))
                    model_state1 = RidgeClassifierCV(alphas=np.array([0.01,0.1,1,10,50,100,500]))
                    model_avg = RidgeClassifier()

                    # train standard model
                    model_standard.fit(X[tot_train_trials,:,t],y[tot_train_trials,t])
                    model_avg.fit(X[tot_train_trials,:,t],y[tot_train_trials,t])
                    estimated_betas_standard[:,t,f] = model_standard.coef_
                    #intercept_standard[t,f] = model_standard.intercept_

                    # test standard model on all
                    y_standard_pred = model_standard.predict(X[tot_test_trials,:,t])

                    # accuracy here
                    acc_standard[t, f] = accuracy_score(y[tot_test_trials, t], y_standard_pred)

                    # test standard model on each state
                    y_standard_pred_state0 = model_standard.predict(X[tot_test_trials_state0, :, t])
                    acc_standard_per_state[0,t,f] = accuracy_score(y[tot_test_trials_state0,t],y_standard_pred_state0)
                    y_standard_pred_state1 = model_standard.predict(X[tot_test_trials_state1, :, t])
                    acc_standard_per_state[1, t, f] = accuracy_score(y[tot_test_trials_state1, t], y_standard_pred_state1)

                    # train per state models
                    model_state0.fit(X[tot_train_trials_state0, :, t], y[tot_train_trials_state0, t])
                    estimated_betas_states[0,:, t, f] = model_state0.coef_
                    #intercept_states[0,t,f] = model_state0.intercept_
                    model_state1.fit(X[tot_train_trials_state1, :, t], y[tot_train_trials_state1, t])
                    estimated_betas_states[1, :, t, f] = model_state1.coef_
                    #intercept_states[1, t, f] = model_state1.intercept_

                    model_avg.coef_ = (model_state1.coef_ + model_state0.coef_) / 2
                    #model_avg.intercept_ = (model_state1.intercept_ + model_state0.intercept_) / 2
                    y_avg_pred = model_avg.predict(X[tot_test_trials, :, t])
                    acc_tot_avg[t, f] = accuracy_score(y[tot_test_trials, t], y_avg_pred)
                    # test model states within a state
                    y_state0_pred_state0 = model_state0.predict(X[tot_test_trials_state0,:,t])
                    acc_per_state[0,t,f] = accuracy_score(y[tot_test_trials_state0,t],y_state0_pred_state0)
                    y_state1_pred_state1 = model_state1.predict(X[tot_test_trials_state1,:,t])
                    acc_per_state[1, t, f] = accuracy_score(y[tot_test_trials_state1, t], y_state1_pred_state1)

                    # test state models across state
                    y_state0_pred_state1 = model_state0.predict(X[tot_test_trials_state1,:,t])
                    acc_cross_state[1,t,f] = accuracy_score(y[tot_test_trials_state1, t],y_state0_pred_state1)
                    y_state1_pred_state0 = model_state1.predict(X[tot_test_trials_state0,:,t])
                    acc_cross_state[0, t, f] = accuracy_score(y[tot_test_trials_state0, t], y_state1_pred_state0)

                    # save all predictions
                    y_pred_standard_all[t].extend(y_standard_pred)
                    y_true_standard_all[t].extend(y[tot_test_trials,t])
                    y_pred_standard_state0[t].extend(y_standard_pred_state0)
                    y_pred_standard_state1[t].extend(y_standard_pred_state1)
                    y_true_state0[t].extend(y[tot_test_trials_state0,t])
                    y_pred_state0[t].extend(y_state0_pred_state0)
                    y_true_state1[t].extend(y[tot_test_trials_state1,t])
                    y_pred_state1[t].extend(y_state1_pred_state1)
                    y_pred_cross_state01[t].extend(y_state0_pred_state1)
                    y_pred_cross_state10[t].extend(y_state1_pred_state0)

                acc_standard_general[t] = accuracy_score(y_true_standard_all[t],y_pred_standard_all[t])
                y_true_state_all = np.concatenate((y_true_state0[t],y_true_state1[t]),0)
                y_pred_state_all = np.concatenate((y_pred_state0[t],y_pred_state1[t]),0)
                acc_per_state_general[t] = accuracy_score(y_true_state_all,y_pred_state_all)
                y_cross_state_all = np.concatenate((y_pred_cross_state10[t], y_pred_cross_state01[t]), 0)

                acc_cross_state_general[t] = accuracy_score(y_true_state_all,y_cross_state_all)

                #acc_standard_general[t] = accuracy_score()


        output_name = "only_mov_standardised_%s_units_per_state_decoding_%s-nbootstrap_%d_state.pkl" % (model_name, decode, n_bootstrap)

        result_rat_dict[mouse_name] = {"acc_standard": acc_standard,'acc_standard_per_state':acc_standard_per_state,
                                       'acc_per_state': acc_per_state,'acc_cross_state': acc_cross_state,
                                       'acc_tot_avg': acc_tot_avg,
                                       'acc_standard_general': acc_standard_general,
                                       'acc_per_state_general': acc_per_state_general,
                                       'acc_cross_state_general':acc_cross_state_general,
                                       'estimated_betas_standard':estimated_betas_standard,
                                       'estimated_betas_states': estimated_betas_states,
                                       'intercept_standard': intercept_standard,
                                       'intercept_states': intercept_states}

        with open(os.path.join(results_folder, output_name), 'wb') as fp:
            pickle.dump(result_rat_dict[mouse_name], fp, protocol=pickle.HIGHEST_PROTOCOL)

        print('%s - Analysis finished and results stored - %s' % (mouse_name, freq_band))


        mean_acc_standard, interval_acc_standard = compute_CI(result_rat_dict[mouse_name]['acc_standard'], axis=1)
        mean_acc_standard_state0, interval_acc_standard_state0 = compute_CI(result_rat_dict[mouse_name]['acc_standard_per_state'][0,:,:], axis=1)
        mean_acc_standard_state1, interval_acc_standard_state1 = compute_CI(result_rat_dict[mouse_name]['acc_standard_per_state'][1,:,:], axis=1)
        mean_acc_state0, interval_acc_state0 = compute_CI(result_rat_dict[mouse_name]['acc_per_state'][0, :, :], axis=1)
        mean_acc_state1, interval_acc_state1 = compute_CI(result_rat_dict[mouse_name]['acc_per_state'][1, :, :], axis=1)
        mean_acc_cross_state01, interval_acc_cross_state01 = compute_CI(result_rat_dict[mouse_name]['acc_cross_state'][1, :, :], axis=1)
        mean_acc_cross_state10, interval_acc_cross_state10 = compute_CI(result_rat_dict[mouse_name]['acc_cross_state'][0, :, :], axis=1)
        mean_acc_tot_avg, interval_acc_tot_avg = compute_CI(result_rat_dict[mouse_name]['acc_tot_avg'], axis=1)

        # plot decoding results with CI
        fig, ax = plt.subplots()
        time_scale = np.arange(0, n_time_points)
        ax.plot(time_scale, mean_acc_standard, color='navy', label='standard')
        ax.fill_between(time_scale, interval_acc_standard[0], interval_acc_standard[1], alpha=0.5,color='navy')
        #ax.plot(time_scale, mean_acc_tot_avg, color='steelblue', label='avg_tot')
        #ax.fill_between(time_scale, interval_acc_tot_avg[0], interval_acc_tot_avg[1], alpha=0.5, color='steelblue')
        ax.plot(time_scale, mean_acc_state0, color='crimson', label='y1,beta1')
        ax.fill_between(time_scale, interval_acc_state0[0], interval_acc_state0[1], alpha=0.5,color='crimson')
        ax.plot(time_scale, mean_acc_state1, color='orange', label='y2,beta2')
        ax.fill_between(time_scale, interval_acc_state1[0], interval_acc_state1[1], alpha=0.5, color='orange')

        #ax.plot(time_scale, mean_acc_standard_state0, color='steelblue', label='y1,beta0')
        #ax.fill_between(time_scale, interval_acc_standard_state0[0], interval_acc_standard_state0[1], alpha=0.5,color='steelblue')
        #ax.plot(time_scale, mean_acc_standard_state1, color='green', label='y2,beta0')
        #ax.fill_between(time_scale, interval_acc_standard_state1[0], interval_acc_standard_state1[1], alpha=0.5,color='green')

        #ax.plot(time_scale, mean_acc_cross_state01, color='pink', label='y2,beta1')
        #ax.fill_between(time_scale, interval_acc_cross_state01[0], interval_acc_cross_state01[1], alpha=0.5, color='pink')
        #ax.plot(time_scale, mean_acc_cross_state10, color='violet', label='y1,beta2')
        #ax.fill_between(time_scale, interval_acc_cross_state10[0], interval_acc_cross_state10[1], alpha=0.5, color='violet')

        ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
        ax.vlines(501, 0, 1, 'k', linestyles='dashdot')
        plt.legend()
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
        ax.set_xlabel('t [s]')
        ax.set_xlim([250, 1001])
        ax.set_ylim([0.25, 1.05])
        ax.set_ylabel('accuracy')
        plt.title('Within and cross decoding, decoding success, %s, %s' % (mouse_name, freq_band))
        plt.show()


        # plot aggregated decoding results
        fig, ax = plt.subplots()
        time_scale = np.arange(0, n_time_points)
        ax.plot(time_scale,result_rat_dict[mouse_name]['acc_standard_general'], color='navy', label='standard')
        ax.plot(time_scale, result_rat_dict[mouse_name]['acc_per_state_general'], color='orangered', label='per-state')
        ax.plot(time_scale, result_rat_dict[mouse_name]['acc_cross_state_general'], color='purple', label='cross-state')
        ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
        ax.vlines(501, 0, 1, 'k', linestyles='dashdot')
        plt.legend()
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
        ax.set_xlabel('t [s]')
        ax.set_xlim([250, 1001])
        ax.set_ylim([0.25, 1.05])
        ax.set_ylabel('accuracy')
        plt.title('Aggregated measures, decoding success, %s , %s' % (mouse_name, freq_band))
        plt.show()



# ----------------------------------------------------------------------------------------------------
# ----------------------- after permutation analysis compute pvalue ----------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------- check bonferroni without any window ------------------------

n_bootstrap = 2
model_name = 'RidgeClassifierCV'
decode = 'success'
n_states = 2
lags = 7
n_time_points=751

result_perm_dict = {}

for mouse_name in ['Buchanan','Stella','Mitt']:
    print(mouse_name)
    if mouse_name == 'Mitt':
        dirdiag = 10000000
    else:
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
    output_name = "standardised_randomised_10kperm_%s_units_per_state_decoding_%s_below100-nbootstrap_%d_state.pkl" % (model_name, decode, n_bootstrap)

    with open(os.path.join(results_folder,output_name), 'rb') as fp:
        result_perm_dict[mouse_name] = pickle.load(fp)

    ref_acc = result_perm_dict[mouse_name]['acc_standard']
    perm_count = 10000 - np.sum(ref_acc==0,1)

    fig = plt.figure()
    plt.plot(np.arange(751),perm_count)
    plt.title('valid perms, %s, min perm nbr %d'%(mouse_name,np.min(perm_count)))
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

# ----------------------------------------------------------------------------------------------
# ----------------------------- window correction -----------------------------------------
# ------------------------------ fill in the missing random ----------------------------------

n_time_points = 751
win_len = 2
win_skip = 2
time_scale = np.arange(n_time_points)
for mouse_name in ['Buchanan','Stella','Mitt']:
    n_time_points = 751
    time_scale = np.arange(n_time_points)

    ref_acc = result_perm_dict[mouse_name]['acc_standard']
    perm_count = 10001 - np.sum(ref_acc == 0, 1)


    #if mouse_name == 'Buchanan':
    #    min_perm_nbr = 2000
    #else:
    #    min_perm_nbr = 1000
    #min_perm_nbr=10000
    #if mouse_name == 'Mitt':
    #    HMM_diff_curve = low_perm_res_dict[mouse_name]['acc_per_state_general'][:, 0] - \
    #                         low_perm_res_dict[mouse_name]['acc_cross_state_general'][:, 0]
    #    hmm_per_state = low_perm_res_dict[mouse_name]['acc_per_state_general'][:, 0]
    #    hmm_cross_state = low_perm_res_dict[mouse_name]['acc_cross_state_general'][:, 0]
    #    time_points_center = np.arange(200 - int(win_len / 2), int(680 - win_len / 2), win_skip)
    #    new_pvalues = np.ones(shape=(len(time_points_center),))

    #else:
    # this can be nan
    result_perm_dict[mouse_name]['acc_per_state_general'][result_perm_dict[mouse_name]['acc_per_state_general']==0]=np.nan
    result_perm_dict[mouse_name]['acc_cross_state_general'][
        result_perm_dict[mouse_name]['acc_cross_state_general'] == 0] = np.nan
    hmm_per_state = result_perm_dict[mouse_name]['acc_per_state_general'][:, 0]
    hmm_cross_state = result_perm_dict[mouse_name]['acc_cross_state_general'][:, 0]
    HMM_diff_curve = hmm_per_state - hmm_cross_state

    time_points_center = np.arange(int(win_len / 2), int(n_time_points - win_len / 2), win_skip)
    new_pvalues = np.ones(shape=(len(time_points_center),))

    # here fill in the missing permutations
    # for each time point to consider, take the average over a window across time points
    i = 0
    for t in time_points_center:

        win_rnd = np.empty(shape=(win_len, 10000))
        win_rnd[:,:] = np.nan

        # if this is nan, the rest has to be nan too
        avg_win_hmm = np.nanmean(HMM_diff_curve[t - int(win_len / 2): t + int(win_len / 2)], 0)

        # find the minimum number of permutations within this window.
        min_perm_nbr_win = np.min(perm_count[t - int(win_len / 2): t + int(win_len / 2)])
        # if it's 0, skip time point

        #if math.isnan(avg_win_hmm):
        if min_perm_nbr_win == 0:
            new_pvalues[i]=1
            i += 1
        else:
        #if min_perm_nbr_win>=100:
            # get all the timepoints with the right perms
            time_slice = np.arange(t - int(win_len / 2), t + int(win_len / 2))
            random_diff_curve = result_perm_dict[mouse_name]['acc_per_state_general'][time_slice, 1:min_perm_nbr_win + 1] - \
                            result_perm_dict[mouse_name]['acc_cross_state_general'][time_slice, 1:min_perm_nbr_win + 1]
            win_rnd[:, :min_perm_nbr_win] = random_diff_curve
            # for each missing permutation, fill it in with randomly picked permutations values
            for k in range(min_perm_nbr_win, 10000):
                win_indices = np.random.randint(1, min_perm_nbr_win, win_len)
                win_rnd[:, k] = random_diff_curve[np.arange(win_len), win_indices]
                # take these indices and fill in the missing result perm dict
                # do not cross time points , otherwise you could add more nans
                result_perm_dict[mouse_name]['acc_per_state_general'][time_slice,1+k] = result_perm_dict[mouse_name]['acc_per_state_general'][time_slice,win_indices]
                result_perm_dict[mouse_name]['acc_cross_state_general'][time_slice,1+k] = result_perm_dict[mouse_name]['acc_cross_state_general'][time_slice,win_indices]

            avg_win_rnd = np.nanmean(win_rnd, 0)
            # TODO: add 1 in the numerator and denominator
            new_pvalues[i] = (np.nansum(avg_win_hmm - avg_win_rnd <= 0) +1)/ 10001
            i += 1

    # here after going through the whole time points we should have everything filled
    # now compute the original pvalues and correct on the whole timeseries
    r_diff_null = result_perm_dict[mouse_name]['acc_per_state_general'][:,1:] - result_perm_dict[mouse_name]['acc_cross_state_general'][:,1:]
    r_diff_observed = result_perm_dict[mouse_name]['acc_per_state_general'][:,0] - result_perm_dict[mouse_name]['acc_cross_state_general'][:,0]
    # TODO: debug how to compute pvalue for all timepoints
    all_time_pvalues = (np.nansum(r_diff_observed[:,np.newaxis] - r_diff_null <= 0, axis=1)+np.ones(shape=(r_diff_observed.shape)))/10001

    corrected_bonferroni, significant_bonferroni = statistics.pval_correction({'pval':all_time_pvalues}, method='bonferroni', alpha=0.05)
    corrected_bhy, significant_bhy = statistics.pval_correction({'pval': all_time_pvalues},method='fdr_by', alpha=0.05)
    corrected_bh, significant_bh = statistics.pval_correction({'pval': new_pvalues}, method='fdr_bh', alpha=0.05)

    # do only for the timepoints available
    # do pval clusterbased correction
    corrected_cluster_based = statistics.pval_cluster_based_correction(test_statistics=r_diff_observed[:,np.newaxis] - r_diff_null, pval=all_time_pvalues)

    #new_pvalues[new_pvalues == 0] = np.nan
    #r_p = {'pval':new_pvalues}
    #pval_corrected_stand, rejected_corrected = statistics.pval_correction(r_p, method='fdr_bh', alpha=0.05)
    #pval_corrected_stand_01, rejected_corrected = statistics.pval_correction(r_p, method='fdr_bh', alpha=0.01)

    result_perm_dict[mouse_name]['acc_per_state_general'][
        result_perm_dict[mouse_name]['acc_per_state_general'] == 0] = np.nan
    result_perm_dict[mouse_name]['acc_cross_state_general'][
        result_perm_dict[mouse_name]['acc_cross_state_general'] == 0] = np.nan

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(time_scale, all_time_pvalues < 0.05, '*g')
    ax.plot(time_scale, all_time_pvalues < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    # ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, significance uncorrected %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(time_points_center, new_pvalues < 0.05, '*g')
    ax.plot(time_points_center, new_pvalues < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    # ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, time-averaged  %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(time_points_center, corrected_bh < 0.05, '*g')
    ax.plot(time_points_center, corrected_bh < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    # ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, time-averaged BH corrected %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(np.arange(250,751), corrected_bhy < 0.05, '*g')
    ax.plot(np.arange(250,751), corrected_bhy < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    #ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    #ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    #ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, significance BHY %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(np.arange(250,751), corrected_cluster_based < 0.05, '*g')
    ax.plot(np.arange(250,751), corrected_cluster_based < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    #ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    #ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    #ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, significance cluster-based %s' % mouse_name)
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(np.arange(n_time_points),hmm_per_cross_diff,color='orangered', label='HMM states')
    # plt.plot(np.arange(n_time_points),np.mean(perm_per_cross_diff,1),color='orchid', label='random states')
    ax.plot(np.arange(250,751), corrected_bonferroni < 0.05, '*g')
    ax.plot(np.arange(250,751), corrected_bonferroni < 0.01, '*k')
    ax.plot(time_scale, hmm_per_state, color='darkorange',
            label='HMM_per_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_per_state_general'][:, 1:1000], axis=1),
    #        color='orchid', label='perm_per_state')

    ax.plot(time_scale, hmm_cross_state, color='purple',
            label='HMM_cross_state')
    # ax.plot(time_scale, np.mean(result_perm_dict[mouse_name]['acc_cross_state_general'][:, 1:1000], axis=1),
    #        color='pink', label='perm_cross_state')

    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(251, 0, 1, 'k', linestyles='dashdot')
    # ax.vlines(484 + 250, 0, 1, 'k', linestyles='dashdot')
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([0, 751])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')

    plt.title('decoding data below 100Hz, significance bonferroni %s' % mouse_name)
    plt.show()


# -----------------------------------------------------------------------------------------------------------------
# ---------------------------------------- show weights ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# show the weights of the state decoders for each subject, for a significant and not significant time point


for mouse_name in mouse_name_list:
    # get the state decoders
    state_weights = np.mean(result_rat_dict[mouse_name]['estimated_betas_states'],3)
    # find timepoint between 250 and 750 where there is the least and the most difference between per-state and cross-state accuracy
    acc_diff = result_rat_dict[mouse_name]['acc_per_state_general'][250:] - result_rat_dict[mouse_name]['acc_cross_state_general'][250:]
    irrelevant_timepoint = np.nanargmin(np.abs(acc_diff))
    significant_timepoint = np.nanargmax(acc_diff)

    fig = plt.figure()
    plt.scatter(state_weights[0,:,irrelevant_timepoint+250],state_weights[1,:,irrelevant_timepoint+250], c='gray')
    #xrange = np.linspace(np.min(state_weights[0,:,irrelevant_timepoint+250]), np.max(state_weights[0,:,irrelevant_timepoint+250]))
    xrange = np.linspace(-0.2,0.2)
    plt.plot(xrange,xrange,'k')
    plt.title('weights for irrelevant point t=%d, rat %s'%(irrelevant_timepoint+250, mouse_name))
    plt.show()

    fig = plt.figure()
    plt.scatter(state_weights[0, :, significant_timepoint + 250], state_weights[1, :, significant_timepoint + 250], c='darkgreen')
    #xrange = np.linspace(np.min(state_weights[0,:,significant_timepoint+250]), np.max(state_weights[0,:,significant_timepoint+250]))
    plt.plot(xrange,xrange,'k')
    plt.title('weights for significant point t=%d, rat %s' % (significant_timepoint + 250, mouse_name))
    plt.show()


# ----------------------------------------------------------------------------------------------------------
# ------------------- -predict states from movement --------------------------------------------------
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LinearRegression, RidgeClassifier
from glhmm import statistics
from Decoding_analysis.Fortin_data_analysis.utils.functions import *
from Decoding_analysis.Fortin_data_analysis.utils.preproc import *
from itertools import permutations, combinations
import os

#----------------- gammas --------------------------------------------
# HMM gammas specifications
# gammas at response have -500 and +500 around response
# gammas at stimulus onset are -250 to +750 around stimulus onset
superchris_bad_trials = [0, 1, 4, 10, 137]

# modulations of
data_aligned_strings = 'response'
decoding_strings = ['success']  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap = 200
min_trial_nbr = 4
n_perms = 1000

result_rat_dict = {}
mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']

for mouse_name in mouse_name_list:
    print(mouse_name)
    n_states=2
    lags=7
    if mouse_name == 'Mitt':
        dirdiag = 10000000
    else:
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name,
                                          'review_analyses')
    gammas_response_name = 'pca_%s_vp_aligned_pokeout_%s_lag%d_k%d_dirdiag%d.mat' % (
            mouse_name, 'orig', lags, n_states, dirdiag)

    info_data_folder = os.path.join(
                '/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                mouse_name)
    info_file = '%s_trial_info.npy' % mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))
    # load movement trials data
    movement_trials = scipy.io.loadmat(os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses',
                     'trials_to_discard.mat'))
    movement_trials_response = movement_trials['trials_locomotion'][0] - 1

    info_data = np.delete(info_data, movement_trials_response, 0)

    success_labels = info_data[:, 0]
    inseq_labels = info_data[:, 1]
    odor_labels = info_data[:, 3]

    gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
    states_around_response = gammas_trials_response['states_around_pokeout_%s' % 'orig']
    states_around_response = np.delete(states_around_response, movement_trials_response,0)

    n_trials, n_time_points = states_around_response.shape
    # load movement
    movement_filename = 'movement_trials_pokeout_aligned.mat'

    movement_data = scipy.io.loadmat(os.path.join(results_folder, movement_filename))
    XY_pos = movement_data['move_around_pokeout']
    XY_vel = movement_data['vel_around_pokeout']
    XY_pos = np.delete(XY_pos,movement_trials_response,0)
    XY_vel = np.delete(XY_vel, movement_trials_response, 0)
    XY_pos[:,0,:] = (XY_pos[:,0,:] -np.mean(XY_pos[:,0,:],1)[:, np.newaxis])/ np.std(XY_pos[:, 0, :],1)[:, np.newaxis]
    XY_pos[:, 1, :] = (XY_pos[:, 1, :] - np.mean(XY_pos[:,1,:],1)[:, np.newaxis])/ np.std(XY_pos[:, 1, :],1)[:, np.newaxis]

    XY_vel[:, 0,:] = (XY_vel[:, 0,:] -np.mean(XY_vel[:, 0,:], 1)[:, np.newaxis]) / np.std(XY_vel[:, 0, :], 1)[:, np.newaxis]
    XY_vel[:, 1, :] = (XY_vel[:, 1, :] - np.mean(XY_vel[:, 1, :], 1)[:, np.newaxis]) / np.std(XY_vel[:, 1, :], 1)[:, np.newaxis]

    # compute confidence interval and plot it neatly for each rat
    mean_xpos, interval_xpos= compute_CI(XY_pos[:, 0, :], axis=0)
    mean_ypos, interval_ypos = compute_CI(XY_pos[:, 1, :], axis=0)
    """
    # plot decoding results with CI
    fig, ax = plt.subplots(nrows=2)
    time_scale = np.arange(0, n_time_points)
    ax[0].plot(time_scale, mean_xpos, color='k')
    ax[0].fill_between(time_scale, interval_xpos[0], interval_xpos[1], alpha=0.5, color='darkgray')
    ax[0].vlines(501, 0, 1, 'k', linestyles='dashdot')
    ax[0].set_xticks([0, 250, 500, 750, 1000])
    #ax[0].set_xticklabels(['-2', '-1', '0', '1', '2'])
    #ax[0].set_xlabel('t [s]')
    ax[0].set_xlim([250, 1001])
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel('x coordinate')
    ax[1].plot(time_scale, mean_ypos, color='k')
    ax[1].fill_between(time_scale, interval_ypos[0], interval_ypos[1], alpha=0.5, color='darkgray')
    ax[1].vlines(501, 0, 1, 'k', linestyles='dashdot')
    ax[1].set_xticks([0, 250, 500, 750, 1000])
    ax[1].set_xticklabels(['-2', '-1', '0', '1', '2'])
    ax[1].set_xlabel('t [s]')
    ax[1].set_xlim([250, 1001])
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel('y coordinate')
    plt.suptitle('Average position of animal head during task, %s' % mouse_name)
    plt.show()
    

    y = states_around_response-1

    accuracy_state_position_response = run_balanced_classification_timevarying_label(XY_pos, y, model_name=model_name,
                                                                                     n_bootstrap=n_bootstrap,
                                                                                     accuracy_metric='accuracy',
                                                                                     return_error=False,
                                                                                     return_coefficients=False)

    # predict also from the velocity and from acceleration
    
    accuracy_state_velocity_response = run_balanced_classification_timevarying_label(XY_vel, y[:, 1:], model_name=model_name,
                                                                                     n_bootstrap=n_bootstrap,
                                                                                     accuracy_metric='accuracy',
                                                                                     return_error=False,
                                                                                     return_coefficients=False)

    output_name = "decoding_states_from_movement.pkl"

    result_rat_dict[mouse_name] = {"acc_pos": accuracy_state_position_response,
                                   'acc_vel': accuracy_state_velocity_response}

    with open(os.path.join(results_folder, output_name), 'wb') as fp:
        pickle.dump(result_rat_dict[mouse_name], fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('%s - Analysis finished and results stored' % mouse_name)


    mean_acc_pos, interval_acc_pos = compute_CI(accuracy_state_position_response, axis=1)
    mean_acc_vel, interval_acc_vel= compute_CI(accuracy_state_velocity_response, axis=1)

    # plot decoding results with CI
    fig, ax = plt.subplots()
    time_scale = np.arange(0, n_time_points)
    ax.plot(time_scale, mean_acc_pos, color='darkgreen', label='position')
    ax.fill_between(time_scale, interval_acc_pos[0], interval_acc_pos[1], alpha=0.5,color='darkgreen')
    ax.plot(time_scale[1:], mean_acc_vel, color='maroon', label='velocity')
    ax.fill_between(time_scale[1:], interval_acc_vel[0], interval_acc_vel[1], alpha=0.5, color='maroon')
    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(501, 0, 1, 'k', linestyles='dashdot')

    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([250, 1001])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')
    plt.title('accuracy predicting states from position and velocity, %s' % mouse_name)
    plt.show()
    """


    y = success_labels

    accuracy_position_success_response = run_balanced_timepoint_classification(XY_pos, y, model_name=model_name,
                                                                                     n_bootstrap=n_bootstrap,
                                                                                     accuracy_metric='accuracy',
                                                                                     return_error=False,
                                                                                     return_coefficients=False)

    accuracy_velocity_success_response = run_balanced_timepoint_classification(XY_vel, y, model_name=model_name,
                                                                                     n_bootstrap=n_bootstrap,
                                                                                     accuracy_metric='accuracy',
                                                                                     return_error=False,
                                                                                     return_coefficients=False)

    output_name = "decoding_success_from_movement.pkl"

    result_rat_dict[mouse_name] = {"acc_pos": accuracy_position_success_response,
                                   'acc_vel': accuracy_velocity_success_response}

    with open(os.path.join(results_folder, output_name), 'wb') as fp:
        pickle.dump(result_rat_dict[mouse_name], fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('%s - Analysis finished and results stored' % mouse_name)

    mean_acc_pos, interval_acc_pos = compute_CI(accuracy_position_success_response, axis=1)
    mean_acc_vel, interval_acc_vel = compute_CI(accuracy_velocity_success_response, axis=1)

    # plot decoding results with CI
    fig, ax = plt.subplots()
    time_scale = np.arange(0, n_time_points)
    ax.plot(time_scale, mean_acc_pos, color='darkgreen', label='position')
    ax.fill_between(time_scale, interval_acc_pos[0], interval_acc_pos[1], alpha=0.5, color='darkgreen')
    ax.plot(time_scale, mean_acc_vel, color='maroon', label='velocity')
    ax.fill_between(time_scale, interval_acc_vel[0], interval_acc_vel[1], alpha=0.5, color='maroon')
    ax.plot(time_scale, np.ones(shape=(n_time_points,)) * 0.5, '--')
    ax.vlines(501, 0, 1, 'k', linestyles='dashdot')

    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
    ax.set_xlabel('t [s]')
    ax.set_xlim([250, 1001])
    ax.set_ylim([0.25, 1.05])
    ax.set_ylabel('accuracy')
    plt.title('accuracy predicting success from position and velocity, %s' % mouse_name)
    plt.show()

# ----------------------------------------------------------------------------------------------------------
# ------------------- -predict labels from movement --------------------------------------------------
