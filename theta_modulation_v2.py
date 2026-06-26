import numpy as np
import pickle
from scipy.io import savemat

def generate_data(n_visits=30, total_dur=300, fs=250,
                  states=((7,0), (9,0)),          # (carrier Hz, phase-mod mean)
                  transition_sec=0.5,             # half-width per boundary; full ramp = 2*transition_sec
                  floor_sec=.5,                  # visit duration floor
                  bf=.97, sigma_f=.1,
                  ba=1e-10, sigma_a=1e-10,
                  bm=.95, sigma_m=.05,
                  m_clip=(-.99, .99),
                  sigma_x=.1,
                  phi0=0.0, a0=None,
                  rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # alternate states
    n_states = states.shape[0]
    ordered_conds = np.repeat(np.arange(n_states), n_visits // n_states)
    n_visits = len(ordered_conds)

    conds = np.full(n_visits, -100)
    prev = -1
    for i in range(n_visits):
        ix = rng.integers(len(ordered_conds))
        s = ordered_conds[ix]

        if i < n_visits - 1:
            while s == prev:
                ix = rng.integers(len(ordered_conds))
                s = ordered_conds[ix]
        
        conds[i] = s
        ordered_conds = np.delete(ordered_conds, ix)
        prev = s

    # durations with floor
    d = rng.dirichlet(np.ones(n_visits))
    durations = np.maximum(total_dur * d, floor_sec)
    durations *= total_dur / durations.sum()

    # samples per visit and total length
    segN = np.maximum(1, np.round(durations * fs).astype(int))
    N = int(segN.sum())

    # visit-wise anchors
    base_f = states[conds, 0].astype(float)
    base_m = states[conds, 1].astype(float)

    # per-sample targets
    mu_f = np.empty(N, float)
    mu_m = np.empty(N, float)
    W = int(round(transition_sec * fs))  # half-width
    idx = 0
    for k in range(n_visits):
        L = segN[k]
        f0, m0 = base_f[k], base_m[k]

        # steady part
        L_steady = L if k == n_visits-1 else max(0, L - 2*W)
        mu_f[idx:idx+L_steady] = f0
        mu_m[idx:idx+L_steady] = m0
        idx += L_steady

        rem = L - L_steady
        if rem <= 0:
            continue

        # transition to next visit’s anchors across 2W (or 'rem' if shorter)
        f1, m1 = base_f[k+1], base_m[k+1]
        wlen = min(rem, 2*W)
        r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, wlen)))
        mu_f[idx:idx+wlen] = (1 - r) * f0 + r * f1
        mu_m[idx:idx+wlen] = (1 - r) * m0 + r * m1
        idx += wlen

    # AR(1) deviations around targets
    delta_f = np.empty(N)
    delta_f[0] = rng.normal(0, sigma_f/np.sqrt(max(1e-12, 1-bf**2)))
    eps_f = rng.normal(0, sigma_f, size=N)

    delta_m = np.empty(N)
    delta_m[0] = rng.normal(0, sigma_m/np.sqrt(max(1e-12, 1-bm**2)))
    eps_m = rng.normal(0, sigma_m, size=N)

    for i in range(1, N):
        delta_f[i] = bf * delta_f[i-1] + eps_f[i]
        delta_m[i] = bm * delta_m[i-1] + eps_m[i]

    f_inst = mu_f + delta_f
    m = np.clip(mu_m + delta_m, m_clip[0], m_clip[1])

    # amplitude envelope
    a = np.empty(N)
    a_mu = 1
    a[0] = a_mu if a0 is None else a0
    eps_a = rng.normal(0, sigma_a, size=N)
    for i in range(1, N):
        a[i] = a_mu + ba * a[i-1] + eps_a[i]

    # phase and signal
    phi = np.empty(N)
    phi[0] = phi0
    for i in range(1, N):
        omega = (2*np.pi * f_inst[i]) / fs
        phi[i] = phi[i-1] + omega * (1 + m[i-1] * np.cos(phi[i-1]))
    x = a * np.sin(phi) + rng.normal(0, sigma_x, size=N)

    out = {
        'x': x,
        'f': f_inst,
        'a': a,
        'phi': phi,
        'm': m,
        'fs': fs,
        'states': np.asarray(states, float),
        'transition_sec': transition_sec
    }
    return out


def plot_signal(dat, zoom_sec=6):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    x = dat['x']
    fs = float(dat['fs'])
    phi = dat['phi']
    f_hz = dat['f']     # instantaneous frequency from generator (Hz)
    a = dat['a']
    m = dat['m']
    transition_sec = dat['transition_sec']

    N = x.size
    t = np.arange(N) / fs

    # instantaneous frequency from phase derivative (empirical)
    dphi = np.full(phi.shape, np.nan)
    dphi[1:] = (phi[1:] - phi[:-1]) * fs
    f_from_phi = dphi / (2*np.pi)  # Hz

    # spectrogram
    nper = int(round(2*fs))  # 2 s window
    nover = int(round(1.5*fs))
    f_spec, t_spec, Sxx = spectrogram(x, fs=fs, nperseg=nper, noverlap=nover, detrend=False)

    # zoom indices: detect first transition
    lag = int(transition_sec * fs * 2)
    f_dif = np.abs(f_hz[lag:] - f_hz[:-lag])
    first_transition = np.where(f_dif[lag:] > np.percentile(f_dif, 99.5))[0][0] + lag
    z0 = int(max(0, first_transition - zoom_sec / 2 * fs))
    z1 = int(min(N, first_transition + zoom_sec / 2 * fs))
    
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), constrained_layout=True)
    # 1) signal
    axes[0].plot(t, x, lw=0.8)
    axes[0].set_title('Signal x(t)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('a.u.')

    # 2) instantaneous frequency: generator vs empirical
    axes[1].plot(t, f_hz, lw=1.0, label='freq (generator)')
    axes[1].plot(t, f_from_phi, lw=0.6, alpha=0.8, label='IF from dφ/dt')
    axes[1].set_title('Instantaneous frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Hz')
    axes[1].legend(loc='upper right')

    # 3) amplitude envelope
    axes[2].plot(t, a, lw=0.8)
    axes[2].set_title('Amplitude a(t)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('a.u.')

    # 4) phase modulation
    axes[3].plot(t, m, lw=0.8)
    axes[3].set_title('Phase-speed modulation m(t)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('a.u.')

    # 6) spectrogram
    im = axes[4].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-12), shading='auto')
    axes[4].set_title('Spectrogram')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Hz')
    axes[4].set_ylim(0, fs/2)
    fig.colorbar(im, ax=axes[4], label='Power (dB)')

    # 7) zoomed panel: signal, freq, phase modulation (scaled)
    axes[5].plot(t[z0:z1], x[z0:z1], lw=0.8, label='signal')
    #axes[5].plot(t[z0:z1], phi_wrapped[z0:z1], lw=0.8, alpha=0.8, label='φ wrapped (rad)')
    axes[5].plot(t[z0:z1], f_hz[z0:z1], lw=0.8, alpha=0.8, label='freq')
    axes[5].plot(t[z0:z1], m[z0:z1] * 5, lw=0.8, alpha=0.9, label='phase mod (scaled)')
    axes[5].axvline(first_transition / fs)
    axes[5].axvline(first_transition / fs - transition_sec)
    axes[5].set_title(f'Zoom ({t[z0]:.2f}–{t[z1-1]:.2f} s)')
    axes[5].set_xlabel('Time (s)')
    axes[5].legend(loc='upper right')

    plt.show()
    return fig


if __name__ == '__main__':

    out_dir = '/home/administrator/hippocampus_Cooper_Fortin/theta_simulations'

    states_list = [((7,0), (9,0)), ((7,0), (7,.6)), ((7,.6), (9,0)), ((6,0), (9,0), (6,.7), (9,.7))]
    states_labels = ['freq2-shape1', 'freq1-shape2', 'freq2-shape2', 'freq2-shape2_independent']

    for states, label in zip(states_list, states_labels):
        data = generate_data(states=np.array(states))

        #with open(f'{out_dir}/data_{label}.pkl', 'wb') as f:
        #    pickle.dump(data, f)
        savemat(f'{out_dir}/data_{label}.mat', data)
        fig = plot_signal(data)
        #fig.savefig(f'{out_dir}/data_{label}.png')
