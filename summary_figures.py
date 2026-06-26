import scipy.io
import numpy as np
import matplotlib.pyplot as plt


filename = '/home/administrator/hippocampus_Cooper_Fortin/results/all_power_spectra.mat'
spectra_data = scipy.io.loadmat(filename)
spectra_data.keys()

x_unsq = np.squeeze(spectra_data['x_all'])
y_unsq = spectra_data['y_all']
y=[]
x=[]
for i in range(5):
    y1=np.squeeze(y_unsq[i,0])
    y2=np.squeeze(y_unsq[i,1])
    y.append((y1,y2))
    x.append(np.squeeze(x_unsq[i]))

y=np.array(y)

linestyles= ['-','--',':','-.']

fig = plt.figure()
for i in range(4):
    plt.plot(x[i],np.squeeze(y[i,0]),color='gold',linewidth=1.5, alpha=0.6, linestyle=linestyles[i])
    plt.plot(x[i],np.squeeze(y[i,1]),color='red',linewidth=1.5, alpha=0.6, linestyle=linestyles[i])

plt.plot(x[i],np.mean(y[:,0,:],axis=0), color='orange', linewidth=2)
plt.plot(x[i],np.mean(y[:,1,:],axis=0), color='crimson', linewidth=2)
#plt.plot(x[4],np.squeeze(y[4,0]),color='purple',linewidth=1.5, alpha=0.5)
#plt.plot(x[4],np.squeeze(y[4,1]),color='green',linewidth=1.5, alpha=0.5)

plt.xlim([0,20])
plt.show()


# ---------------------------------------------------------------------------------------
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


filename = '/home/administrator/hippocampus_Cooper_Fortin/results/all_gamma_lifetimes.mat'
lifetime_data = scipy.io.loadmat(filename)
lifetime_data.keys()

x_unsq = lifetime_data['x_all']
y_unsq = lifetime_data['y_all']
y=[]
x=[]
for i in range(5):
    y1=np.squeeze(y_unsq[i,0])
    y2=np.squeeze(y_unsq[i,1])
    y.append((y1,y2))
    x.append(np.squeeze(x_unsq[i,0]))

y=np.array(y)
x=np.array(x)

fig = plt.figure()
for i in range(4):
    plt.plot(x[i],np.squeeze(y[i,0]),color='orange',linewidth=1.5, alpha=0.6, linestyle=linestyles[i])
    plt.plot(x[i],np.squeeze(y[i,1]),color='red',linewidth=1.5, alpha=0.5, linestyle=linestyles[i])

plt.plot(x[i],np.mean(y[:,0,:],axis=0), color='orange', linewidth=2)
plt.plot(x[i],np.mean(y[:,1,:],axis=0), color='crimson', linewidth=2)
#plt.plot(x[4],np.squeeze(y[4,0]),color='purple',linewidth=1.5, alpha=0.7)
#plt.plot(x[4],np.squeeze(y[4,1]),color='green',linewidth=1.5, alpha=0.7)
#plt.xlim([0,45])
plt.show()


# make x into Hz
# x is timepoints.
# each 250 timepoints is a sec
# divide x by 250
# then use 1/x to find Hz


fig = plt.figure()
for i in range(4):
    x_time = x[i][1:] / 250
    print(x_time)
    plt.plot(1/x_time,np.squeeze(y[i,0][1:]),color='orange',linewidth=1.5, alpha=0.6, linestyle=linestyles[i])
    plt.plot(1/x_time,np.squeeze(y[i,1][1:]),color='red',linewidth=1.5, alpha=0.5, linestyle=linestyles[i])

#plt.plot(x[i],np.mean(y[:,0,:],axis=0), color='orange', linewidth=2)
#plt.plot(x[i],np.mean(y[:,1,:],axis=0), color='crimson', linewidth=2)
#plt.plot(x[4],np.squeeze(y[4,0]),color='purple',linewidth=1.5, alpha=0.7)
#plt.plot(x[4],np.squeeze(y[4,1]),color='green',linewidth=1.5, alpha=0.7)
#plt.xlim([0,45])
plt.title('frequency?')
plt.show()





# --------------------------------------------------------------------------------------

filename = '/home/administrator/hippocampus_Cooper_Fortin/results/all_gamma_periodicity.mat'
gamma_period_data = scipy.io.loadmat(filename)
gamma_period_data.keys()

x_unsq = np.squeeze(gamma_period_data['x_all'])
y_unsq = np.squeeze(gamma_period_data['y_all'])
y=[]
x=[]

for i in range(5):
    y.append(np.squeeze(y_unsq[i]))
    x.append(np.squeeze(x_unsq[i]))

y = np.array(y)
x = np.array(x)
fig = plt.figure()
for i in range(4):
    plt.plot(x[i],y[i],color='gray',linewidth=1.5, alpha=0.5, linestyle=linestyles[i])

plt.plot(x[i],np.mean(y,axis=0), color='black', linewidth=2)
#plt.plot(x[i],np.mean(y[:,1,:],axis=0), color='crimson', linewidth=2.5)#
#plt.plot(x[4],np.squeeze(y[4]),color='black',linewidth=1.5, alpha=0.7)
#plt.plot(x[4],np.squeeze(y[4,1]),color='green',linewidth=1.5, alpha=0.2)

plt.show()


# ---------------------------------------------------------------------------------
# plot FO box

FO_state1 = [40,44,49,40]
FO_state2 = [60,56,51,60]

fig = plt.figure()
bplot = plt.boxplot([FO_state1,FO_state2], patch_artist=True, positions=[1,2], widths=0.6)
plt.ylim([0,101])
for patch,color in zip(bplot['boxes'],['crimson','orange']):
    patch.set_facecolor(color)

plt.show()



# -------------------------------------------------------------------------
# states lifetime
from Decoding_analysis.Fortin_data_analysis.utils.functions import *
import scipy.stats

# --------------------------------- gammas --------------------------------------------
# HMM gammas specifications
# gammas at response have -500 and +500 around response
# gammas at stimulus onset are -250 to +750 around stimulus onset
superchris_bad_trials = [0,1,4,10,137]

# modulations of
data_aligned_strings = 'response'
decoding_strings = ['success']  # odor needs to be done differently
model_name = 'RidgeClassifierCV'
n_bootstrap = 25
min_trial_nbr = 5

result_rat_dict = {}
mouse_name_list =['Buchanan', 'Stella', 'Mitt', 'Barat', 'Superchris']
bs_stored_prediction = {}
bs_stored_prediction_units = {}
bs_stored_true_label_units = {}
for mouse_name in mouse_name_list:
    print(mouse_name)
    if mouse_name == 'Mitt':
        gamma_directory = 'gamma_k2_l7_d10m'
        # gammas
        n_states = 2
        lags = 7
        dirdiag = 10000000
    else:
        gamma_directory = 'gamma_k2_l7_d1bn'
        # gammas
        n_states = 2
        lags = 7
        dirdiag = 1000000000

    results_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/results/',mouse_name,'review_analyses')
    gammas_response_name = 'pca_%s_vp_aligned_pokeout_orig_lag%d_k%d_dirdiag%d.mat' % (mouse_name, lags, n_states, dirdiag)

    info_data_folder = os.path.join('/home/administrator/hippocampus_Cooper_Fortin/pre-processed_data/stattensor/',mouse_name)
    info_file = '%s_trial_info.npy'%mouse_name.lower()
    info_data = np.load(os.path.join(info_data_folder, info_file))
    if mouse_name=='Superchris':
        info_data = np.delete(info_data, superchris_bad_trials,0)
    success_labels = info_data[:,0]
    inseq_labels = info_data[:,1]
    odor_labels = info_data[:,3]

    units_density_folder = '/home/administrator/hippocampus_Cooper_Fortin/results/%s'%mouse_name
    units_density_response_name = 'units_density_all_kernels_response_aligned.pkl'

    with open(os.path.join(units_density_folder, units_density_response_name), 'rb') as fp:
        units_density_response = pickle.load(fp)

    gammas_trials_response = scipy.io.loadmat(os.path.join(results_folder, gammas_response_name))
    states_around_response = gammas_trials_response['states_around_pokeout_orig']

    # -------------------------------- retrieve useful info -------------------------------------
    n_trials, n_units, n_time_points, n_kernels = units_density_response.shape
    # choose kernel 1
    dk = 1

    # ----------------------------- get the gammas shape from vpath -----------------------------
    # convert vp to gammas shape
    gamma_vp_response = convert_vp_to_gamma(states_around_response, states_change=None)

    # compute average states life time
    state_lifetime_1 = []
    state_lifetime_2 = []
    for i in range(n_trials):
        count_1 = 0
        count_2 = 0
        for j in range(1,n_time_points):
            if states_around_response[i, j] == states_around_response[i, j - 1]:
                if states_around_response[i,j] == 1:
                    count_1 +=1
                else:
                    count_2 +=1
            else:
                if states_around_response[i,j-1] == 1:
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
    plt.title('states lifetime, %s'%mouse_name)
    plt.ylabel('count')
    plt.legend()
    plt.xlabel('active time points')
    plt.show()

