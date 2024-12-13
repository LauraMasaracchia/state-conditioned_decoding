""" Here are analysis functions for the State-conditioned decoding project.

Author: Laura Masaracchia, laurama@cfin.au.dk
"""

import numpy as np
import scipy
from scipy.signal import windows
from glhmm.auxiliary import padGamma
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as st


def hmm_spectral_multitaper(data, T, Gamma=None, options=None):
    """
    Compute multitaper spectral measures:
    - power,
    - coherence,
    - corss power spectral density

    Parameters:
    - data: np.ndarray or list of np.ndarrays (Trials concatenated)
    of dimensions n_samples (total n_timepoints) by n_channels
    - T: np.ndarray or list (lengths of each trial) with dimensions n_trials (this script assumes one subject / session)
    - Gamma: np.ndarray (state time courses), with dimensions n_samples by K (n_states),
    if not, pad gamma  if Gamma is none, it means no states.
    The function will compute the spectral properties of the signal assuming one state
    - options: dict containing options:
        - 'Fs' : sampling rate (COMPULSORY)
        - 'order' or 'embeddedlags' : (COMPULSORY IF HMM TRAINED WITH THOSE)
        - 'fpass' : frequency range for the power spectrum estimation

        for the multitapers
        - 'win' : window length
        - 'tapers_res' : half time bandwidth
        - 'n_tapers' : number of tapers

        preprocessing
        - 'standardize' : if to standardize data

    Returns:
    - fit: dict with spectral estimates per state
    for each state there is a dict containing:
        - 'f' : frequency bins, with shape n_freq
        - 'p' : power spectrum, with shape n_freq by n_channels
        - 'psdc' : cross-channel power spectral density, with shape n_freq by n_channels by n_channels
        - 'coh' : channels coherence, with shape n_freq by n_channels by n_channels


    Author: Laura Masaracchia
    Email: laurama@cfin.au.dk
    Date: 3/12/2024

    """

    # --------- basic checks ------------
    if options is None:
        raise AssertionError(
            'options has to be specified. For option fields like Fs there is no fdefault and they cannot be inferred from the data')

        # if data and T are list, concatenate
    if isinstance(data, list):
        TT = np.concatenate([np.array(t).flatten() for t in T])
        data = [np.array(d) for d in data]
    else:
        TT = np.array(T).flatten()
        data = np.array(data)

    T = TT

    # check data dimensions
    if len(data.shape) > 2:
        raise AssertionError(
            'Data dimensions incompatible with analysis. Data should be of shape n_samples, n_channels')
    if len(data.shape) == 1:  # one channel
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape

    # check T
    if int(np.sum(T)) != n_samples:
        raise AssertionError('T must be either a list or a numpy array with entries the length of each trial')

    # check Gamma
    if Gamma is None:
        Gamma = np.ones([n_samples, 1], dtype=np.float32)

    K = Gamma.shape[1]

    # if length of Gamma is different than n_samples, pad gamma
    if Gamma.shape[0] != n_samples:
        if ('order' in options.keys()) or ('embeddedlags' in options.keys()):
            Gamma = padGamma(Gamma, T, options)
        else:
            raise AssertionError('Wrong shape of Gamma, order or embeddedlags need to be specified in the options')

    # -------------- retrieve parameters from options -------------
    # Sampling frequency
    if 'Fs' in options.keys():
        Fs = options['Fs']
    else:
        raise AssertionError('Sampling frequency Fs needs to be specified in options')

    # Frequency range to be considered
    if 'fpass' in options.keys():
        fpass = options['fpass']
        if len(fpass) != 2:
            print('WARNING: fpass field in option mispecified. Continuing with default range, [0, Fs/2]')
            fpass = [0, Fs / 2]
    else:  # default, from 0 to nyquist frequency
        fpass = [0, Fs / 2]

    if 'win_len' in options.keys():
        window_length = int(options['win_len'])
    else:  # default: Fs * 2
        window_length = int(Fs * 2)

    if 'n_tapers' in options.keys():
        n_tapers = options['n_tapers']
    else:  # set default value to 5 (OSL dynamics default is 7). this is typically NW*2-1
        n_tapers = 5

    if 'tapers_res' in options.keys():
        time_half_bandwidth = options['tapers_res']
    else:  # set default to 3 (OSL dynamics has it to 4)
        time_half_bandwidth = 3

    # standardise data for the computation of the power spectra
    if 'standardize' in options.keys():
        if options['standardize'] == 1:
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Initialize fit dictionary
    fit = {}

    # ---------------- compute frequency grid and tapers ----------------

    # number of fft should be power of 2 to speed up computations
    nfft = 2 ** int(np.ceil(np.log2(window_length)))

    # Frequency grid
    f, findx = getfgrid(Fs, nfft, fpass)
    n_freq = len(f)

    # compute tapers
    dpss_tapers = windows.dpss(window_length, time_half_bandwidth, n_tapers)
    # dpss_tapers has shape [n_tapers, window_length]
    # check that it has the right shape
    # for debugging purposes
    # print(dpss_tapers.shape)

    # get scaling coefficient to account for states activation period
    scaling_coefs = n_samples / (np.sum(Gamma ** 2, axis=0))

    for k in range(K):

        # multiply data by the states probability
        X = data * Gamma[:, k][:, np.newaxis]

        # compute state-spectra
        p, psdc = multitaper_spectra(X, dpss_tapers, window_length, nfft, findx)

        # Scaling for the multitapers
        p *= 2 / Fs
        psdc *= 2 / Fs

        # average over time windows
        p = np.mean(p, axis=0)
        psdc = np.mean(psdc, axis=0)

        # scaling for the Gamma
        p *= scaling_coefs[k]
        psdc *= scaling_coefs[k]

        # calculate coherence
        #coh = multitaper_coherence(psdc)

        if n_channels == 1:
            # squeeze eventual extra dimensions
            p = np.squeeze(p)
            psdc = np.squeeze(psdc)
            #coh = np.squeeze(coh)

        # store results for each state
        fit[k] = {'f': f,  # frequency
                  'p': p,  # power
                  'psdc': psdc,  # cross-power spectral density
                  #'coh': coh,  # coherence
                  }

    return fit


# ----------------------- functions ---------------

def getfgrid(Fs, nfft, fpass):
    """
    Generate frequency grid for FFT computation.
    """
    df = Fs / nfft
    f = np.linspace(0, Fs - df, nfft)
    findx = (f >= fpass[0]) & (f <= fpass[1])
    return f[findx], findx


def multitaper_spectra(data, tapers, window_length, nfft, findx):
    """
    Perform multitaper spectral analysis on input data.

    Parameters:
    - data: 2D array with dimensions n_samples,n_channels.
    - tapers: multitapers, with dimensions n_tapers, window_length.
    - window_length: integers.
    - nfft: int, number of FFT points.
    - findx: indices of frequency to keep (frequency range for power spectrum estimation)


    Returns:
    - PW: power spectrum, dimensions n_windows, n_channels, n_freq
    - CPW: cross-channels power spectrum, with dimensions n_windows, n_channels, n_channels, n_freq
    """

    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape
    n_freq = np.sum(findx * 1)

    # pad data to have enough timepoints for the windows
    # pad only along time axis, keep channels unchanged
    data = np.pad(data, window_length // 2)[:, window_length // 2: n_channels + window_length // 2]

    # get number of windows
    n_windows = n_samples // window_length

    # initialize p
    PW = np.empty(shape=(n_windows, n_channels, n_freq))
    # inizialize cross p
    CPW = np.empty([n_windows, n_channels, n_channels, n_freq], dtype=np.complex64)

    for w in range(n_windows):
        # select data within window,
        # transpose to make data compatible with tapers multiplicaton
        x_window = data[w * window_length:(w + 1) * window_length, :].transpose(1, 0)

        # check dimensions for debugging purposes
        NC, WLD = x_window.shape  # channels, window_length
        NT, WLT = tapers.shape  # n_tapers, window_length
        if WLD != WLT:
            raise AssertionError('data and tapers dimensions are not compatible')

        # multiply data with tapers
        # add dimension for tapers to the data,
        # add dimension for channels to the tapers
        tapered_x = x_window[np.newaxis, :, :] * tapers[:, np.newaxis, :]
        # shape n_tapers,n_channels,window_length

        # compute fourier transform (along time axis)
        taper_fft = np.fft.fft(tapered_x, nfft)
        taper_range = taper_fft[:, :, findx]
        # should be of shape [n_tapers, n_channels,n_freq]

        # compute power
        power_window = np.real(np.conj(taper_range) * taper_range)
        # compute cross-power
        cross_power_window = np.conj(taper_range)[:, :, np.newaxis, :] * taper_range[:, np.newaxis, :, :]

        # average spectra across tapers
        PW[w, :, :] = np.mean(power_window, axis=0)
        CPW[w, :, :, :] = np.mean(cross_power_window, axis=0)

    return PW, CPW


def convert_vp_to_gamma(vp, states_change=None):
    n_trials,n_time_points = vp.shape
    if states_change is None:
        n_states = len(np.unique(vp))
        gamma_vp = np.zeros(shape=(n_trials, n_states, n_time_points))

        for t in range(n_time_points):
            for k in range(n_states):
                state_trials = np.where(vp[:,t] == k+1)[0]
                gamma_vp[state_trials, k, t] = 1

    else:
        raise AssertionError('n this version it is not possible to change states, i.e. merge or discard')

    return gamma_vp


def run_class_balanced_decoding(X, y, n_rep=100, return_pred=False, return_all=False):

    accuracy_pred = np.empty(shape=(n_rep,))
    X_test_all = {}
    y_true_all = {}
    y_pred_all = {}
    model_all = {}

    # class0 and class1 trials
    class0_trials = np.where(y == 0)[0]
    class1_trials = np.where(y == 1)[0]
    train_nbr = int(np.min((len(class0_trials), len(class1_trials))) * 0.8)
    test_nbr = int(np.min((len(class0_trials), len(class1_trials))) * 0.2)

    # the prediction mirrors exactly the distribution of y, so we need to do it in a class-balanced fashion
    # predict the state at each time point
    for b in range(n_rep):

        np.random.shuffle(class0_trials)
        np.random.shuffle(class1_trials)
        train_set_state1 = class0_trials[:train_nbr]
        test_set_state1 = class0_trials[train_nbr:train_nbr + test_nbr]
        train_set_state2 = class1_trials[:train_nbr]
        test_set_state2 = class1_trials[train_nbr:train_nbr + test_nbr]

        X_train = np.concatenate((X[train_set_state1], X[train_set_state2]), axis=0)
        y_train = np.concatenate((y[train_set_state1], y[train_set_state2]), axis=0)
        X_test = np.concatenate((X[test_set_state1], X[test_set_state2]), axis=0)
        y_test = np.concatenate((y[test_set_state1], y[test_set_state2]), axis=0)

        model = RidgeClassifier(alpha=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_pred[b] = accuracy_score(y_test, y_pred)

        if return_all:
            X_test_all[b] = X_test
            y_true_all[b] = y_test
            y_pred_all[b] = y_pred
            model_all[b] = model

    if return_all:
        return (accuracy_pred, X_test_all, y_true_all, y_pred_all, model_all)
    else:
        return accuracy_pred


def compute_CI(x, confidence=0.95, axis=0):

    x_mean = np.mean(x, axis)
    CI = st.norm.interval(confidence=confidence,loc=x_mean,scale=st.sem(x, axis=axis))
    return x_mean, CI


def sort_trials_by_variables(variable1,variable2):
    # assuming both variables are binary
    # if variable 1 is success and fail -> cond1 is 1 and cond 2 is 0
    # and variable 2 is state -> cond1 is A (state 1) and cond2 is B (state 2)

    var1_cond1_trials = np.where(variable1==0)[0]
    var1_cond2_trials = np.where(variable1==1)[0]
    var2_cond1_trials = np.where(variable2 == 0)[0]
    var2_cond2_trials = np.where(variable2 == 1)[0]

    # find intersection
    var1_cond1_var2_cond2_trials = list(set(var1_cond1_trials) & set(var2_cond2_trials))
    var1_cond1_var2_cond1_trials = list(set(var1_cond1_trials) & set(var2_cond1_trials))
    var1_cond2_var2_cond2_trials = list(set(var1_cond2_trials) & set(var2_cond2_trials))
    var1_cond2_var2_cond1_trials = list(set(var1_cond2_trials) & set(var2_cond1_trials))

    # spit them as:
    # 0A , 0B
    # 1A , 1B
    sorted_trials = [[var1_cond1_var2_cond1_trials, var1_cond1_var2_cond2_trials],[var1_cond2_var2_cond1_trials, var1_cond2_var2_cond2_trials]]
    return sorted_trials


def group_trials_variable(sorted_trials, axis):
    # spit out the intersection of trials grouped along one axis
    # if axis=0, returns them grouped by row (behaviour)
    # if axis=1, returns them grouped by column (state)
    if axis==0:
        group1 = set(sorted_trials[0][0]) | set(sorted_trials[0][1])
        group2 = set(sorted_trials[1][0]) | set(sorted_trials[1][1])
    elif axis==1:
        group1 = set(sorted_trials[0][0]) | set(sorted_trials[1][0])
        group2 = set(sorted_trials[0][1]) | set(sorted_trials[1][1])
    else:
        raise AssertionError('Not implemented yet')

    return list(group1), list(group2)

