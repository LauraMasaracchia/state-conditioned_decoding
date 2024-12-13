"""
This script performs all the preprocessing steps on the spike data.
On the raw spike data, the following steps are performed, in order:
1. compute spike density
2. Downsample to 250 Hz (original sampling frequency is 1kHz)
3. Cut data into trials

The script should be run per rat.
It outputs a '[rat_name]_spike_density_G10ms_250Hz_response.pkl' file with the preprocessed data

Author: Laura Masaracchia, laurama@cfin.au.dk
"""


import scipy.io
import scipy.signal
from elephant.statistics import instantaneous_rate
from quantities import ms
from neo import SpikeTrain
from elephant.kernels import GaussianKernel
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from .utils.analysis import run_class_balanced_decoding, compute_CI


# ----------------------------------------------------------------------------
# mouse_name_list = ['Buchanan','Stella','Mitt','Barat','Superchris']
# ---------------------------- select rat name here --------------------------
rat_name = 'Buchanan'

# ----------------------------------------------------------------------------
# ---------------------------- write paths to data ---------------------------
input_folder_name = './data/continuous_data/'
output_folder_name = './data/preprocessed_data/'
output_filename = '%s_spike_density_G10ms_250Hz_response.pkl' % rat_name
results_directory = './results/'

spike_file = '%s_spike_data_binned.npy'%rat_name.lower()
filename_behav_info = 'behav_info.pkl'

# ----------------------------------------------------------------------------
# -------------------------------- load data ---------------------------------

# load behavioural info
with open(os.path.join(output_folder_name, rat_name, filename_behav_info), 'rb') as fp:
    behav_info = pickle.load(fp)

# load spike data
units_raw = scipy.io.loadmat(os.path.join(input_folder_name, rat_name, spike_file))

# downsample data
resampled_units = scipy.signal.decimate(units_raw['ensemble'][:,1:-1], 4)

poke_out_time = behav_info['poke_out_time']
success_label = behav_info['success_label']
n_trials = len(success_label)

n_time_points = 1001

reshape_resampled = np.transpose(resampled_units, (1, 0))
n_units = reshape_resampled.shape[0]

# cut data into trials
trial_spikes_response = np.zeros(shape=(n_trials, n_units, n_time_points))
for i in range(n_trials):
    trial_spikes_response[i, :, :] = reshape_resampled[:, poke_out_time[i,0]-501:poke_out_time[i,0]+500]

# --------------------- compute spike densities ----------------------------------------------------------------------
# decide on window size for a smoothing gaussian kernel
kernel_size = [10, 20, 40] # compute various kernels and test
n_kernels = len(kernel_size)

units_density_response = np.zeros(shape=(n_trials, n_units, n_time_points, n_kernels))
# create and store spike density
for j in range(n_trials):
    for i in range(n_units):
        spike_indices = np.where(trial_spikes_response[j, i, :] > 0)[0]
        spike_train = SpikeTrain(spike_indices * ms, t_stop=n_time_points)
        for k in range(n_kernels):
            spike_density_tmp = instantaneous_rate(spike_train, sampling_period=1 * ms,
                                                   kernel=GaussianKernel(kernel_size[k] * ms))
            units_density_response[j, i, :, k] = spike_density_tmp.ravel()


# -------------------------------------------------------------
# plot some units to see effect of smoothing kernel
for k in range(len(kernel_size)):
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.6)
    for i in range(5):
        for j in range(10):
            axs[i].plot(np.arange(n_time_points), units_density_response[j,i,:,k])
        axs[i].set_title("unit %d"%i)
        axs[i].set_xlabel('t [ms]')
        axs[i].set_ylabel('spike density')

    fig.suptitle("units density only 10 trials, kernel=%d "%kernel_size[k])
    plt.show()


# -------------------------------- save all units density computed -------------------------------
# decided for kernel 1
with open(os.path.join(output_folder_name, rat_name, output_filename), 'wb') as fp:
    pickle.dump(units_density_response[:,:,:,1], fp, protocol=pickle.HIGHEST_PROTOCOL)


# perform class-balanced standard decoding here and save results

# standardise units per time point and per trial, across neurons
n_rep = 500
X_units = units_density_response[:,:,:,1]
X_units = (X_units - np.mean(X_units, 1)[:, np.newaxis, :]) / np.std(X_units, 1)[:, np.newaxis, :]

accuracy_units_pred = np.empty(n_time_points,n_rep)

for t in range(n_time_points):
    y_t = success_label
    X_t = X_units[:, :, t]
    accuracy_units_pred[t,:] = run_class_balanced_decoding(X=X_t,
                                                           y=y_t,
                                                           n_rep=n_rep)

# compute confidence interval
mean_acc, interval_acc = compute_CI(accuracy_units_pred, axis=1)


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
plt.title('%s prediction of trial outcome from standardised neural activity' % rat_name)
plt.show()


# -----------------------------------------------------------------------------------
# save results
pred_acc_filename = '%s_predict_trial_outcome_from_units_rep%d.pkl'%(rat_name,n_rep)
with open(os.path.join(results_directory, rat_name, pred_acc_filename), 'wb') as fp:
    pickle.dump(accuracy_units_pred, fp, protocol=pickle.HIGHEST_PROTOCOL)


