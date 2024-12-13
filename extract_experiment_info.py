"""
This script should be used to extract behavioural information needed for further analysis, from the original data.
It outputs a file behav_info.pkl for the selected rat.

Author: Laura Masaracchia, laurama@cfin.au.dk
"""

import scipy
import os
import pickle
import numpy as np
import glob

# select rat name among: 'Buchanan', 'Stella', 'Mitt', 'Barat', 'Superchris'
rat_name = 'Buchanan'

# file with behavioral details
input_folder_name = './data/continuous_data/'
filename_behav = '.*_BehaviorMatrix.mat'

output_folder_name = './data/preprocessed_data/'
filename_output = 'behav_info.pkl'

# load data
behav_file = glob.glob(os.path.join(input_folder_name,rat_name,filename_behav))
behav = scipy.io.loadmat(behav_file[0])
behav_data = behav['behavMatrix']

# extract useful info
poke_events = behav_data[:,13]
trialPerf = behav_data[:,12]
trialInSeqLog = behav_data[:,11]

# find poke timings (poke out = response)
poke_in_time = np.where(poke_events==1)[0]
poke_out_time = np.where(poke_events==-1)[0]

# find success label and trial type label
succ_idx = np.where(trialPerf==1)[0]
fail_idx = np.where(trialPerf==-1)[0]
succ_fail_idx = np.concatenate((succ_idx,fail_idx),0)
succ_fail_idx = np.sort(succ_fail_idx)

# success label is with +1 if success, -1 if fail
success_label = trialPerf[succ_fail_idx]

# resample the poke out timings :
# original sampling frequency is 1kHz, new sampling freq is 250 Hz
poke_in_time = np.round(poke_in_time/4).astype('int')
poke_out_time = np.round(poke_out_time/4).astype('int')

n_trials = len(success_label)

info_dict = {'n_trials':n_trials,
             'poke_out_time':poke_out_time,
             'success_label':success_label}
# save info
with open(os.path.join(output_folder_name,rat_name,filename_output), 'wb') as fp:
    pickle.dump(info_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

