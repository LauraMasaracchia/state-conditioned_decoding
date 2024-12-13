""" Here are some functions used for the preprocessing of the raw data.

Author: Laura Masaracchia, laurama@cfin.au.dk
"""

import numpy as np
import scipy.signal


def filter_data(data, Fs, cutoff, filter_type='high', filter_order=3, axis_filter=0, standardise=True):
    """ function to filter the data, using a butter filter, of order 3
    it takes as input:
    COMPULSORY
    - data : the data to be filtered,
    - Fs : the data sampling rate in Hz,
    - cutoff : the cutoff frequency in Hz,

    OPTIONAL
    - filter_type : string, the type of filtering option, default = 'high'. other option is 'low',
    - filter order : the order of the butter filter to use. default = 3,
    - axis_filter : the axis along which to filter the data. default = 0,
    - standardise : boolean, whether to standardise the data. Default is True.

    """

    nyq = Fs // 2
    normal_cutoff = cutoff / nyq
    if standardise:
        data = (data - np.mean(data, axis_filter)) / np.std(data,axis_filter)

    sos = scipy.signal.butter(filter_order, normal_cutoff, btype=filter_type, analog=False, output='sos')

    # Apply the filter
    data_filtered = scipy.signal.sosfiltfilt(sos, data, axis=axis_filter)

    return data_filtered

