
# data exploration of Fortin Lab data

from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LinearRegression, RidgeClassifier
from glhmm import statistics
from Decoding_analysis.Fortin_data_analysis.utils.functions import *
from Decoding_analysis.Fortin_data_analysis.utils.preproc import *
from sklearn.mixture import GaussianMixture
from itertools import permutations, combinations

# ----------------------- perform analyses deconfounding movement ------------------
# ----------------------- we do this in two ways ---------------------------------
# -----------------------------------------------------------------------------

# here we try to visualize the different neural patterns under the two different states
# -----------------------------------------------------------------------------


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


mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']

for mouse_name in mouse_name_list:
    print(mouse_name)
    n_states = 2
    lags = 7
    if mouse_name == 'Mitt':
        dirdiag = 10000000
    else:
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/', mouse_name, 'review_analyses')
    gammas_response_name = 'pca_%s_vp_aligned_pokeout_orig_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)

    info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',
                                        mouse_name)
    info_file = '%s_trial_info.npy' % mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))

    success_labels = info_data[:, 0]

    units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s' % mouse_name
    units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

    with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
        units_density_response = pickle.load(fp)

    # delete the trials also from gammas and unit data
    gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
    states_around_response = gammas_trials_response['states_around_pokeout_orig']

    # -------------------------------- retrieve useful info -------------------------------------

    n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
    # choose kernel 1
    dk = 1

    # ----------------------------- get the gammas shape from vpath -----------------------------
    # convert vp to gammas shape
    gamma = convert_vp_to_gamma(states_around_response, states_change=None)
    n_states = gamma.shape[1]

    X = units_density_response[:, :, :, dk]
    X = (X - np.mean(X, 1)[:, np.newaxis,:]) / np.std(X, 1)[:, np.newaxis,:]

    y = success_labels

    diff_over_time_state0 = np.zeros(shape=(n_units,n_time_points))
    diff_over_time_state1 = np.zeros(shape=(n_units, n_time_points))

    for t in range(n_time_points):
        print('time point t = %d' % t)

        sorted_trials = sort_trials_by_variables(y, gamma[:, 1, t])

        # spitted as:
        # 0A , 0B
        # 1A , 1B

        min_nbr_per_class_state0 = min(len(sorted_trials[0][0]), len(sorted_trials[1][0]))

        # avg activity state A, class 0
        avg_act_A0 = np.mean(X[sorted_trials[0][0],:,t],0)
        avg_act_A1 = np.mean(X[sorted_trials[1][0], :, t], 0)
        avg_act_B0 = np.mean(X[sorted_trials[1][0], :, t], 0)
        avg_act_B1 = np.mean(X[sorted_trials[1][1], :, t], 0)

        diff_over_time_state0[:,t] = avg_act_A0 - avg_act_A1
        diff_over_time_state1[:, t] = avg_act_B0 - avg_act_B1

        # show states weights
    fig, ax = plt.subplots(nrows=3)

    im1 = ax[0].imshow(diff_over_time_state0,cmap='coolwarm', vmin=-1.5, vmax=1.5)
    ax[0].set_title('state 1 encoding differently success vs failure')
    #fig.colorbar(im1, ax=ax[0], location='bottom', orientation='horizontal')
    im2 = ax[1].imshow(diff_over_time_state1,cmap='coolwarm', vmin=-1.5, vmax=1.5)
    ax[1].set_title('state 2 encoding differently success vs failure')
    # ax.set_xticks([0, 250, 500, 750])
    # ax.set_xticklabels(['-1', '0', '1', '2'])
    fig.colorbar(im2, ax=[ax[0],ax[1]], location='right', orientation='vertical')
    im3 = ax[2].imshow(np.abs(diff_over_time_state0 - diff_over_time_state1), cmap='hot', vmin=0, vmax=2)
    fig.colorbar(im3, ax=ax[2], location='right', orientation='vertical')
    ax[2].set_title('absolute difference in activity between states')
    fig.suptitle('activity patterns, %s' % mouse_name)
    plt.show()
