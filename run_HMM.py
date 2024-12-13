"""
Script to run the HMM-TDE on the preprocessed LFP data and perform the states characterization analysis.
The script outputs a '.pkl' file named with the specific details of the HMM run, with the HMM and its outputs.

The HMM used is implemented in the GLHMM toolbox.
The analyses are performed per rat.

The script computes:
 1. the power spectrum of the preprocessed LFP data ,
 2. FO, switching rate and lifetimes of the states,
 3. states power spectra,
 4. periodicity of the states activation,
 5. temporal states distribution along a trial, on average across trials divided by success and failure

Author: Laura Masaracchia
Email: laurama@cfin.au.dk
"""

# imports
import numpy as np
from .utils.analysis import hmm_spectral_multitaper
import os
import matplotlib.pyplot as plt
import pickle
from glhmm import glhmm, utils, preproc, auxiliary, graphics

# ----------------------------------------------------------------------------
# choose rat name = ['Buchanan','Stella','Mitt','Barat','Superchris']
# ---------------------------- select rat name here --------------------------
rat_name = 'Buchanan'

# ----------------------------------------------------------------------------
# ---------------------------- write paths to data ---------------------------
# file with behavioral details
input_folder_name = './data/preprocessed_data/'
LFP_filename = '%s_1st_LFP_pc_filtered_250Hz.pkl' % rat_name
results_directory = './results/'

# ----------------------------------------------------------------------------
# -------------------------------- load data ---------------------------------

with open(os.path.join(input_folder_name, rat_name, LFP_filename), 'rb') as fp:
    X = pickle.load(fp)

# get frequency content of the data

# define useful quantities
Fs = 250  # sampling frequency in Hz
frange = [0.1, 100]
T = [len(X)]

options = {'Fs': Fs, 'fpass': frange}

# get frequency content with multitaper
spectra_global = hmm_spectral_multitaper(X, T, options=options)

# plot data spectra
fig = plt.figure()
plt.plot(spectra_global[0]['f'], spectra_global[0]['p'], 'k')
plt.xlabel('frequency (Hz)')
plt.ylabel('power (a.u.)')
plt.title('LFP data power spectrum, %s' % rat_name)
plt.show()

# ----------------------------------------------------------------------------
# ------------------------------- HMM analysis -------------------------------
# ----------------------------------------------------------------------------

# set HMM analysis configuration
# dirichlet diagonal = 10000000 for Mitt, 1000000000 for all the others
options['dirichlet_diag'] = 1000000000
options['embeddedlags'] = list(range(-7, 8, 1))
options['K'] = 2

options['model_mean'] = 'no'
options['model_beta'] = 'no'
options['covtype'] = 'full'
options['cyc'] = 50

# start and end of sessions indices
# here, one session per rat
idx_data = np.array([[0, len(X)]])

# --------------------------------------------------------------------
# prepare data for HMM,
# make the TDE embeddings on the data
X_embedded, idx_tde = preproc.build_data_tde(X, idx_data, options['embeddedlags'])

# --------------------------------------------------------------------
# initialize HMM
TDE_hmm = glhmm.glhmm(model_beta=options['model_beta'],
                      model_mean=options['model_mean'],
                      K=options['K'],
                      covtype=options['covtype'],
                      dirichlet_diag=options['dirichlet_diag'])

# run HMM
print('Training HMM-TDE, mouse %s' % rat_name)

stc_tde, xi_tde, fe_tde = TDE_hmm.train(X=None, Y=X_embedded, indices=idx_tde, options=options)
vpath_tde = TDE_hmm.decode(X=None, Y=X_embedded, viterbi=True)

paddedGamma = auxiliary.padGamma(stc_tde, T, options=options)
paddedVP = auxiliary.padGamma(vpath_tde, T, options=options)

# plot state time courses with signal
plotting_range = np.arange(24000, 250000)
graphics.plot_vpath(paddedVP[plotting_range], signal=X[plotting_range], title="States and signal example")

# -------------------------------------------------------------------
# inspect states - basic sanity checks
entropy_FO_TDE_hmm = utils.get_FO_entropy(stc_tde, np.array([idx_data]))
cov_matrix_tde = TDE_hmm.get_covariance_matrix()
FO = utils.get_FO(paddedGamma, indices=idx_data)
SR = utils.get_switching_rate(paddedGamma, indices=idx_data)
LTmean, LTmed, LTmax = utils.get_life_times(paddedVP, indices=idx_data)

# plot some relevant statistics
graphics.plot_FO(FO, num_ticks=FO.shape[0])
graphics.plot_switching_rates(SR, num_ticks=SR.shape[0])

# -------------------------------------------------------------------
# get states spectral properties

# Compute multitaper spectrum
spectral_states = hmm_spectral_multitaper(X, T, paddedGamma, options=options)

# -------------------------------------------------------------------
# save results

result_tde_hmm_dict = {'hmm': TDE_hmm,
                       'stc': stc_tde,
                       'paddedGamma': paddedGamma,
                       'paddedVP': paddedVP,
                       'options': options,
                       'spectral_properties': spectral_states}

output_filename = 'hmm_tde_analysis_%s_dd%d_l%d_K%d.pkl' % (
rat_name, options['dirichlet_diag'], options['embeddedlags'], options['K'])
with open(os.path.join(results_directory, rat_name, output_filename), 'wb') as fp:
    pickle.dump(result_tde_hmm_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------------
# -------------------------- states characterization -------------------------
# ----------------------------------------------------------------------------
# load results

# rat name
rat_name = 'Buchanan'

# file with behavioral details
input_folder_name = './data/preprocessed_data/'
filename_behav_info = 'behav_info.pkl'
LFP_filename = '%s_1st_LFP_pc_filtered_250Hz.pkl' % rat_name
results_directory = './results/'

# load results
with open(os.path.join(results_directory, rat_name, output_filename), 'rb') as fp:
    result_tde_hmm_dict = pickle.load(fp)

# load behavioural info
with open(os.path.join(input_folder_name, rat_name, filename_behav_info), 'rb') as fp:
    behav_info = pickle.load(fp)

# load LFP data
with open(os.path.join(input_folder_name, rat_name, LFP_filename), 'rb') as fp:
    X = pickle.load(fp)


spectral_states = result_tde_hmm_dict['spectral_properties']
options = result_tde_hmm_dict['options']

# plot states spectral characteristics
f = spectral_states[0]['f']
p_state1 = spectral_states[0]['p']
p_state2 = spectral_states[1]['p']

# plot power spectrum for each state
fig = plt.figure()
plt.plot(f, p_state1, color='crimson', label='state1')
plt.plot(f, p_state2, color='orange', label='state2')
plt.legend()
plt.title('states power spectrum, %s' % rat_name)
plt.xlabel('frequency (Hz)')
plt.xlim([0, 25])
plt.show()

# plot periodicity of states activation
paddedGamma = result_tde_hmm_dict['paddedGamma']
paddedVP = result_tde_hmm_dict['paddedVP']
# get frequency content with multitaper
state_activation_power_spectrum = hmm_spectral_multitaper(paddedGamma[:,0], T, options=options)
# plot data spectra
fig = plt.figure()
plt.plot(state_activation_power_spectrum[0]['f'], state_activation_power_spectrum [0]['p'], 'k')
plt.xlabel('frequency (Hz)')
plt.ylabel('power (a.u.)')
plt.title('state activation power spectrum, %s' % rat_name)
plt.show()

# check states distribution around response and per variable
poke_out_time = behav_info['poke_out_time']
# success label is with +1 if success, -1 if fail
success_label = behav_info['success_label']
n_trials = len(success_label)
n_states = options['K']
# timepoints total per trial, 2 sec before and 2 sec after response
n_time_points = 1001


# -----------------------------------------------------------------------------------
# ------------------------------ cut data into trials -------------------------------
states_around_pokeout = np.empty(shape=(n_trials, n_time_points, n_states))
lfp_around_pokeout = np.empty(shape=(n_trials, n_time_points))
vp_around_pokeout = np.empty(shape=(n_trials, n_time_points))

for i in range(n_trials):
    states_around_pokeout[i] = paddedGamma[poke_out_time[i] - 500:poke_out_time[i] + 501, :]
    vp_around_pokeout[i] = paddedVP[poke_out_time[i] - 500:poke_out_time[i] + 501]
    lfp_around_pokeout[i] = X[poke_out_time[i] - 500:poke_out_time[i] + 501, :].flatten()

# check states distribution
fig = plt.figure()
plt.plot(np.arange(n_time_points), np.mean(states_around_pokeout, 0)[:, 1], 'r', label='state1')
plt.plot(np.arange(n_time_points), np.mean(states_around_pokeout, 0)[:, 0], 'y', label='state2')
plt.title('states distribution around response')
plt.show()

# -----------------------------------------------------------------------------------
# plot states distribution divided between success and fail

states_success = states_around_pokeout[success_label==1]
states_fail = states_around_pokeout[success_label==-1]

fig,axs = plt.subplots(nrows=2)
axs[0].plot(np.arange(n_time_points),np.mean(states_success, 0)[:, 0], 'r', label='state1')
axs[0].plot(np.arange(n_time_points),np.mean(states_success, 0)[:, 1], 'y', label='state2')
axs[0].vlines(501, 0, 1, 'k', linestyles='dashdot', label='response')
axs[0].set_xticks([0, 250, 500, 750, 1000])
axs[0].set_xticklabels([])
axs[0].set_ylabel('probability')
axs[1].plot(np.arange(n_time_points),np.mean(states_fail, 0)[:, 0], 'r', label='state1')
axs[1].plot(np.arange(n_time_points),np.mean(states_fail, 0)[:, 1], 'y', label='state2')
axs[1].vlines(501, 0, 1, 'k', linestyles='dashdot', label='response')
axs[1].set_xlabel('time (s)')
axs[1].set_xticks([0, 250, 500, 750, 1000])
axs[1].set_xticklabels(['-2', '-1', '0', '1', '2'])
plt.legend()
axs[0].set_title('success')
axs[1].set_title('fail')
fig.suptitle(' temporal states distribution,  %s' % rat_name)
plt.show()

# compute average states life time
state_lifetime_1 = []
state_lifetime_2 = []
for i in range(n_trials):
    count_1 = 0
    count_2 = 0
    for j in range(1, n_time_points):
        if states_around_pokeout[i, j] == states_around_pokeout[i, j - 1]:
            if states_around_pokeout[i, j] == 1:
                count_1 += 1
            else:
                count_2 += 1
        else:
            if states_around_pokeout[i, j - 1] == 1:
                state_lifetime_1.append(count_1)
                count_1 = 0
            else:
                state_lifetime_2.append(count_2)
                count_2 = 0

fig = plt.figure()
counts_1, bins_1 = np.histogram(state_lifetime_1, bins=50)
plt.hist(bins_1[:-1], bins_1, weights=counts_1, color='crimson', label='state1', alpha=0.7)
counts_2, bins_2 = np.histogram(state_lifetime_2, bins=50)
plt.hist(bins_2[:-1], bins_2, weights=counts_2, color='orange', alpha=0.7, label='state2')
plt.title('states lifetime, %s'%rat_name)
plt.ylabel('count')
plt.legend()
plt.xlabel('time points')
plt.show()

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# save state time courses cut into trials
states_trial_resp = {'Gamma': states_around_pokeout,
                     'VP': vp_around_pokeout}

states_around_resp_filename = 'states_around_response_%s_dd%d_l%d_K%d.pkl' % (
    rat_name, options['dirichlet_diag'], options['embeddedlags'][-1], options['K'])
with open(os.path.join(results_directory, rat_name, states_around_resp_filename), 'wb') as fp:
    pickle.dump(states_trial_resp, fp, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------------
# save LFP data cut into trials
lfp_around_resp_filename = '%s_lfp_around_response.pkl'%rat_name
with open(os.path.join(input_folder_name, rat_name, lfp_around_resp_filename), 'wb') as fp:
    pickle.dump(lfp_around_pokeout, fp, protocol=pickle.HIGHEST_PROTOCOL)

