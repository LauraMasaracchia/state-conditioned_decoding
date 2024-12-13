"""
Script to run the simulations on overlapping information scenarios.
This script generates multiple scenarios of two-set synthetic data with varying degrees pf similarity in their
encoding weights. Then uses standard and state-conditioned decoding to decode information
and plots the decoding accuracies as a function of the information overlapping parameter alpha.

Author: Laura Masaracchia
Email: laurama@cfin.au.dk
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

# set simulation parameters similar to our experimental data
# nbr trials = n, nbr neurons = p, mean ans std of the encoding weights, noise
n1 = 100
n2 = 100
p = 50
mean1 = 0
mean2 = 0
std1 = 4
std2 = 4
noise_std1 = 1
noise_std2 = 1

# overlapping parameter alpha
alpha_range = np.arange(0,1.05,0.05)
alpha_len = len(alpha_range)

# cross validation folds
n_folds = 10
# number of repetitions of the analysis for each value of alpha
n_repetitions = 100

# initialize
yacc11 = np.zeros(shape=(alpha_len,n_repetitions,n_folds))
yacc22 = np.zeros(shape=(alpha_len,n_repetitions,n_folds))
ycross12 = np.zeros(shape=(alpha_len,n_repetitions,n_folds))
ycross21 = np.zeros(shape=(alpha_len,n_repetitions,n_folds))

yacc_cross = np.zeros(shape=(alpha_len,n_repetitions,n_folds))
yacc_state = np.zeros(shape=(alpha_len,n_repetitions,n_folds))
yacc_total = np.zeros(shape=(alpha_len,n_repetitions,n_folds))

# ------------------------------- simulations ------------------------------------
for xi in range(alpha_len):
    alpha = alpha_range[xi]
    for g in range(n_repetitions):
        x1 = np.random.random(size=(n1, p))
        x2 = np.random.random(size=(n2, p))
        B1 = np.random.normal(mean1, std1, size=(p,))
        B2 = np.random.normal(mean2, std2, size=(p,))

        beta1 = B1
        beta2 = alpha * B1 + (1-alpha) * B2

        noise1 = np.random.normal(0, noise_std1, size=(n1,))
        noise2 = np.random.normal(0, noise_std2, size=(n2,))

        y1 = np.sign(np.matmul(x1, beta1) + noise1)
        y2 = np.sign(np.matmul(x2, beta2) + noise2)

        indices1 = np.arange(n1)
        indices2 = np.arange(n2)
        kf = KFold(n_splits=n_folds, shuffle=True)
        k = 0

        # cross-validated decoding
        for train_index, test_index in kf.split(indices1):
            x1_train, x1_test = x1[train_index], x1[test_index]
            y1_train, y1_test = y1[train_index], y1[test_index]

            x2_train, x2_test = x2[train_index], x2[test_index]
            y2_train, y2_test = y2[train_index], y2[test_index]

            x_train = np.concatenate((x1_train, x2_train), axis=0)
            x_test = np.concatenate((x1_test, x2_test), axis=0)

            y_train = np.concatenate((y1_train, y2_train), axis=0)
            y_test = np.concatenate((y1_test, y2_test))

            # within-set decoding
            model_1 = RidgeClassifier(alpha=0.01)
            model_1.fit(x1_train, y1_train)

            y1_pred = model_1.predict(x1_test)

            model_2 = RidgeClassifier(alpha=0.01)
            model_2.fit(x2_train, y2_train)

            y2_pred = model_2.predict(x2_test)

            yacc11[xi,g, k] = accuracy_score(y1_test, y1_pred)
            yacc22[xi,g, k] = accuracy_score(y2_test, y2_pred)

            # cross-set decoding
            y1_pred_2 = model_2.predict(x1_test)
            y2_pred_1 = model_1.predict(x2_test)
            ycross21[xi, g, k] = accuracy_score(y1_test, y1_pred_2)
            ycross12[xi, g, k] = accuracy_score(y2_test, y2_pred_1)

            y_cross_pred = np.concatenate((y1_pred_2, y2_pred_1), axis=0)
            y_state_pred = np.concatenate((y1_pred, y2_pred), axis=0)

            yacc_cross[xi, g, k] = accuracy_score(y_test, y_cross_pred)
            yacc_state[xi,g, k] = accuracy_score(y_test, y_state_pred)

            # standard decoding
            model = RidgeClassifier(alpha=0.005)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            yacc_total[xi,g, k] = accuracy_score(y_test, y_pred)

            k += 1


fig = plt.figure()

plt.plot(alpha_range, np.mean(np.mean(yacc11, axis=1),1), color='orange', label='y1,beta1')
plt.plot(alpha_range, np.mean(np.mean(yacc22, axis=1),1), color='crimson', label='y2,beta2')
plt.plot(alpha_range, np.mean(np.mean(ycross21, axis=1),1), color='orchid', label='y1,beta2')
plt.plot(alpha_range, np.mean(np.mean(ycross12, axis=1),1), color='pink', label='y2,beta1')
plt.plot(alpha_range, np.mean(np.mean(yacc_cross, axis=1),1), color='purple', label='y cross')
plt.plot(alpha_range, np.mean(np.mean(yacc_state, axis=1),1), color='orangered', label='y state')
plt.plot(alpha_range, np.mean(np.mean(yacc_total,axis=1),1), color='navy', label='y standard')
plt.legend()
plt.ylim([0.5,1.01])
plt.xlabel('alpha')
plt.ylabel('avg CVEV')
plt.title('Simulations, N=%d, p=%d, repetitions=%d' % (n1, p, n_repetitions))
plt.show()
