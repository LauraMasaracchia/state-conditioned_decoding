"""
This script performs all the preprocessing steps on the LFP data.
On the raw LFP data, the following steps are performed, in order:
1. Principal Component Analysis (PCA) - only the 1st PC is kept
2. High-pass filtering at 4 Hz
3. Downsample to 250 Hz (original sampling frequency is 1kHz)

The script should be run per rat.
It outputs a '[rat_name]_1st_LFP_pc_filtered_250Hz.pkl' file with the preprocessed data

Author: Laura Masaracchia, laurama@cfin.au.dk
"""

# all the imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import scipy.signal
from sklearn.decomposition import PCA
import pickle
from .utils.preprocessing import filter_data


# ----------------------------------------------------------------------------
# mouse_name_list = ['Buchanan','Stella','Mitt','Barat','Superchris']
# ---------------------------- select rat name here --------------------------
rat_name = 'Buchanan'

# ----------------------------------------------------------------------------
# ---------------------------- write paths to data ---------------------------
input_folder_name = './data/continuous_data/'
output_folder_name = './data/preprocessed_data/'
output_filename = '%s_1st_LFP_pc_filtered_250Hz.pkl' % rat_name

# ----------------------------------------------------------------------------
# ---------------------------------- load data -------------------------------
# -------------------------------- LFP data ---------------------------------
# note that different rats have a different name
# Stella and Superchris have '_SM', the others have '_T'
lfp_files = glob.glob(os.path.join(input_folder_name, rat_name, '/*_SM.mat'))
# lfp_files=glob.glob(os.path.join(LFP_parent_directory,mouse_name,'/*_T.mat'))
n_LFP_channels = len(lfp_files)
continuous_LFP = scipy.io.loadmat(lfp_files[0])

n_timepoints = continuous_LFP['statMatrix'].shape[0]
# initialize the data, to be filled.
# raw LFP data for each electrode are saved in column 2
X_full = np.empty(shape=(n_timepoints, n_LFP_channels))
X_full[:,0] = continuous_LFP['statMatrix'][:, 1]

for i in range(1, n_LFP_channels):
    continuous_LFP = scipy.io.loadmat(lfp_files[i])
    # to be checked if it is a dictionary
    X_full[:, i] = continuous_LFP['statMatrix'][:, 1]

# plot signal just to check
# fig = plt.figure()
# for i in range(n_LFP_channels):
#     plt.plot(np.arange(1000),X_full[1000:2000,i]+i)
# plt.show()

# ----------------------------------------------------------------------------
# ----------------------- dimensionality reduction ---------------------------
# pca wants X of dimensions (n_samples, n_features)
pca = PCA()
X_decompose = pca.fit_transform(X_full)

X_1st_pc = X_decompose[:, 0]

# plot explained variance
# fig = plt.figure()
# plt.plot(np.arange(n_LFP_channels),np.cumsum(pca.explained_variance_))
# plt.show()

# ----------------------------------------------------------------------------
# ------------------------------- filter -------------------------------------
X_filtered = filter_data(X_1st_pc, Fs=1000, cutoff=4,
                         filter_type='high',
                         filter_order=3,
                         axis_filter=0,
                         standardise=True)

# ----------------------------------------------------------------------------
# ------------------------------- downsample ---------------------------------
# from 1kHz to 250 Hz, divide by 4
X_downsampled = scipy.signal.decimate(X_filtered, 4)

# ----------------------------------------------------------------------------
# -------------------------------- save data ---------------------------------

with open(os.path.join(output_folder_name, rat_name, output_filename), 'wb') as fp:
    pickle.dump(X_downsampled, fp, protocol=pickle.HIGHEST_PROTOCOL)
