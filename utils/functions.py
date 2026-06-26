
import numpy as np
import scipy.stats as st
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV, ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import Ridge
from scipy.special import comb

# ---------------------- need some class balance -----------------

def compute_valid_perm_number(y,state,kmin=4):
    Ny0 = np.sum(y==0)
    Ny1 = np.sum(y==1)
    Nst1 = np.sum(state==0)
    Nst2 = np.sum(state==1)
    Nmin = np.min([Ny0,Ny1,Nst1,Nst2])
    Nysmall= np.min([Ny0,Ny1])
    Nybig = np.max([Ny0,Ny1])
    Nstmin = np.min([Nst1,Nst2])

    nvalid_perm = 0

    for k in range(kmin,Nmin-kmin+1):
        nvalid_perm += comb(Nysmall,k)*comb(Nybig,Nstmin-k)

    return nvalid_perm




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


# Define some useful functions
def run_balanced_classification_timevarying_label(X, y, model_name='RidgeClassifierCV', n_bootstrap=100, accuracy_metric='accuracy',
                                          return_error=False, return_prediction=False, return_coefficients=False):

    n_trials, n_units, n_time_points = X.shape
    # ------------ create a list of possible folds combining states and labels balance -------------------
    accuracy_stored = np.zeros(shape=(n_time_points, n_bootstrap))
    betas = np.zeros(shape=(n_units,n_time_points,n_bootstrap))
    if return_prediction:
        # initialize tensor with shape #trials, #time points, n_folds,
        # that you fill with a logical value for the right or wrong prediction
        pred_logik = np.empty(shape=(n_time_points,n_trials,n_bootstrap))
        pred_logik[:,:,:] = np.nan

    if len(y.shape) ==1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))
        print('y of dimension 1: repeating it to make it 2 dimensional')
    elif len(y.shape)>2:
        raise AssertionError('wrong shape of y')
    for t in range(n_time_points):
        # this code assumes that y is binary, 2 dimensional and varies at eawch time point
        X_time_point = X[:,:,t]
        y_time_point = y[:,t]

        if np.sum(y_time_point == 1) < np.sum(y_time_point == 0):
            smaller_class_nbr = np.sum(y_time_point == 1)
            smaller_label = 1
            bigger_label = 0
        else:
            smaller_class_nbr = np.sum(y_time_point == 0)
            smaller_label = 0
            bigger_label = 1

        # we will take 80% of these trials for training, and the same amount in the next
        trials_smallest_class = np.where(y_time_point == smaller_label)[0]  # smaller class trials indices
        trials_biggest_class = np.where(y_time_point == bigger_label)[0]  # bigger class trials indices

        # use kfolds only in the smallest batch
        training_nbr = int(0.8 * smaller_class_nbr)
        testing_nbr = int(0.2 * smaller_class_nbr)

        np.random.seed(10)
        print('timepoint t = %d' % t)
        if training_nbr<2:
            print('not enough training trials at time point t = %d. Skip t'%t)
            continue
        else:
            for k in range(n_bootstrap):
                # for every kfold shuffle the order of the trials, get first 80% as training, remaining 10% as testing,
                # same for the bigger class
                np.random.shuffle(trials_smallest_class)
                np.random.shuffle(trials_biggest_class)
                small_training_trials = trials_smallest_class[0:training_nbr]
                small_testing_trials = trials_smallest_class[training_nbr:training_nbr + testing_nbr]
                big_training_trials = trials_biggest_class[0:training_nbr]
                big_testing_trials = trials_biggest_class[training_nbr:training_nbr + testing_nbr]

                train_index = np.concatenate((small_training_trials, big_training_trials))
                np.random.shuffle(train_index)
                test_index = np.concatenate((small_testing_trials, big_testing_trials))
                np.random.shuffle(test_index)

                # split everything into train and test
                # store test indices to use for gammas later

                X_train, X_test = X_time_point[train_index], X_time_point[test_index]
                y_train, y_test = y_time_point[train_index], y_time_point[test_index]

                if model_name == 'RidgeClassifierCV':
                    model = RidgeClassifierCV()
                else:
                    raise AssertionError('only available model RidgeClassifierCV for now')

                model.fit(X_train, y_train)
                # get betas
                betas[:, t, k] = model.coef_
                # predict and compute accuracy
                y_pred = model.predict(X_test)
                if return_prediction:
                    pred_logik[t,test_index,k] = (y_pred == y_test)*1.0

                if accuracy_metric=='accuracy':
                    accuracy_stored[t, k] = accuracy_score(y_test, y_pred)
                else:
                    raise AssertionError('only available accuracy metric is accuracy score for now')

    if return_prediction:
        return (accuracy_stored,pred_logik)
    if return_coefficients:
        return (accuracy_stored, betas)
    else:
        return accuracy_stored


# Define some useful functions
def run_balanced_timepoint_classification(X, y, model_name='RidgeClassifierCV', n_bootstrap=10, accuracy_metric='accuracy',
                                          return_error=False, return_prediction=False, return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # ------ establish the class imbalance -------
    nbr_class_1 = np.sum(y == 1)
    nbr_class_2 = np.sum(y == 0)

    if nbr_class_2 < nbr_class_1:
        smaller_class_nbr = nbr_class_2
        smaller_label = 0
        bigger_label = 1
    else:
        smaller_class_nbr = nbr_class_1
        smaller_label = 1
        bigger_label = 0

    # we will take 90% of these trials for training, and the same amount in the next
    trials_smallest_class = np.where(y == smaller_label)[0]  # smaller class trials indices
    trials_biggest_class = np.where(y == bigger_label)[0]  # bigger class trials indices

    # use kfolds only in the smallest batch
    training_nbr = int(0.8 * smaller_class_nbr)
    testing_nbr = int(0.2 * smaller_class_nbr)

    accuracy_stored = np.zeros(shape=(n_time_points, n_bootstrap))
    trials_test_error = np.zeros(shape=(n_time_points, testing_nbr * 2,n_bootstrap))
    betas = np.zeros(shape=(n_units, n_time_points,n_bootstrap))
    test_indices = {}  # useful to check where the decoder fails later
    y_pred_tot = np.empty(shape=(n_trials, n_time_points,n_bootstrap))
    y_pred_tot[:,:,:] = np.nan
    np.random.seed(10)
    for k in range(n_bootstrap):
        # for every kfold shuffle the order of the trials, get first 90% as training, remaining 10% as testing,
        # same for the bigger class
        np.random.shuffle(trials_smallest_class)
        np.random.shuffle(trials_biggest_class)
        small_training_trials = trials_smallest_class[0:training_nbr]
        small_testing_trials = trials_smallest_class[training_nbr:training_nbr + testing_nbr]
        big_training_trials = trials_biggest_class[0:training_nbr]
        big_testing_trials = trials_biggest_class[training_nbr:training_nbr + testing_nbr]

        train_index = np.concatenate((small_training_trials, big_training_trials))
        np.random.shuffle(train_index)
        test_index = np.concatenate((small_testing_trials, big_testing_trials))
        np.random.shuffle(test_index)

        # split everything into train and test
        # store test indices to use for gammas later
        test_indices[k] = test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train.shape)
        print(y_train.shape)

        # train on one time point
        for t_i in range(n_time_points):
            # create all training test split
            X_train_timepoint = X_train[:, :, t_i]
            X_test_timepoint = X_test[:, :, t_i]
            # make sure that you can handle y multidimension
            if len(y.shape) > 1:
                if model_name == 'RidgeClassifierCV':
                    y_train_timepoint = y_train[:, t_i]
                    y_test_timepoint = y_test[:, t_i]
                else:
                    y_train_timepoint = y_train[:, t_i].astype(float) * 2 - 1
                    y_test_timepoint = y_test[:, t_i].astype(float) * 2 - 1
            else:
                if model_name == 'RidgeClassifierCV':
                    y_train_timepoint = y_train
                    y_test_timepoint = y_test
                else:
                    y_train_timepoint = y_train.astype(float) * 2 - 1
                    y_test_timepoint = y_test.astype(float) * 2 - 1

            # the model has to be instantiated before every use
            if model_name == 'RidgeClassifierCV':
                model = RidgeClassifierCV()
            elif model_name == 'ElasticNetCV':
                model = ElasticNetCV(max_iter=5000, tol=0.01)
            else:
                raise AssertionError("other models not available yet")

            model.fit(X_train_timepoint, y_train_timepoint)
            y_pred_timepoint = model.predict(X_test_timepoint)

            if return_coefficients:
                betas[:, t_i, k] = model.coef_
            if return_prediction:
                y_pred_tot[test_index, t_i, k] = y_pred_timepoint

            trials_test_error[t_i, :, k] = (y_test_timepoint - y_pred_timepoint) ** 2
            if accuracy_metric == "accuracy":
                if model_name == 'RidgeClassifierCV':
                    accuracy_stored[t_i, k] = accuracy_score(y_test_timepoint, y_pred_timepoint)
                else:
                    accuracy_stored[t_i, k] = accuracy_score(y_test_timepoint, np.sign(y_pred_timepoint))
            elif accuracy_metric == "mean_squared_error":
                accuracy_stored[t_i, k] = mean_squared_error(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == "explained_variance":
                accuracy_stored[t_i, k] = explained_variance_score(y_test_timepoint, y_pred_timepoint)

        print("mean overall accuracy fold %d" % k)
        print(np.mean(accuracy_stored[:, k]))
        k += 1

    if return_prediction:
        return (accuracy_stored, y_pred_tot)
    elif return_error:
        return (accuracy_stored, trials_test_error, test_indices)
    elif return_coefficients:
        return (accuracy_stored, betas)
    else:
        return accuracy_stored

# Define some useful functions
def run_balanced_shuffled_timepoint_classification(X, y, model_name='RidgeClassifierCV', n_perms=1000, accuracy_metric='accuracy'):
    n_trials, n_units, n_time_points = X.shape
    # ------ establish the class imbalance -------
    nbr_class_1 = np.sum(y == 1)
    nbr_class_2 = np.sum(y == 0)

    if nbr_class_2 < nbr_class_1:
        smaller_class_nbr = nbr_class_2
        smaller_label = 0
        bigger_label = 1
    else:
        smaller_class_nbr = nbr_class_1
        smaller_label = 1
        bigger_label = 0

    # use kfolds only in the smallest batch
    training_nbr = int(0.8 * smaller_class_nbr)
    testing_nbr = int(0.2 * smaller_class_nbr)

    accuracy_stored = np.zeros(shape=(n_time_points, n_perms))
    # for each permutation simply shuffle all trials, everything else will stay the same
    for g in range(n_perms):
        # we will take 90% of these trials for training, and the same amount in the next
        np.random.shuffle(y)
        trials_smallest_class = np.where(y == smaller_label)[0]  # smaller class trials indices
        trials_biggest_class = np.where(y == bigger_label)[0]  # bigger class trials indices

        # for every kfold shuffle the order of the trials, get first 90% as training, remaining 10% as testing,
        # same for the bigger class
        np.random.shuffle(trials_smallest_class)
        np.random.shuffle(trials_biggest_class)
        small_training_trials = trials_smallest_class[0:training_nbr]
        small_testing_trials = trials_smallest_class[training_nbr:training_nbr + testing_nbr]
        big_training_trials = trials_biggest_class[0:training_nbr]
        big_testing_trials = trials_biggest_class[training_nbr:training_nbr + testing_nbr]

        train_index = np.concatenate((small_training_trials, big_training_trials))
        np.random.shuffle(train_index)
        test_index = np.concatenate((small_testing_trials, big_testing_trials))
        np.random.shuffle(test_index)

        # split everything into train and test
        # store test indices to use for gammas later
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train.shape)
        print(y_train.shape)

        # train on one time point
        for t_i in range(n_time_points):
            # create all training test split
            X_train_timepoint = X_train[:, :, t_i]
            X_test_timepoint = X_test[:, :, t_i]
            # make sure that you can handle y multidimension
            if len(y.shape) > 1:
                if model_name == 'RidgeClassifierCV':
                    y_train_timepoint = y_train[:, t_i]
                    y_test_timepoint = y_test[:, t_i]
                else:
                    y_train_timepoint = y_train[:, t_i].astype(float) * 2 - 1
                    y_test_timepoint = y_test[:, t_i].astype(float) * 2 - 1
            else:
                if model_name == 'RidgeClassifierCV':
                    y_train_timepoint = y_train
                    y_test_timepoint = y_test
                else:
                    y_train_timepoint = y_train.astype(float) * 2 - 1
                    y_test_timepoint = y_test.astype(float) * 2 - 1

            # the model has to be instantiated before every use
            if model_name == 'RidgeClassifierCV':
                model = RidgeClassifierCV()
            elif model_name == 'ElasticNetCV':
                model = ElasticNetCV(max_iter=5000, tol=0.01)
            else:
                raise AssertionError("other models not available yet")

            model.fit(X_train_timepoint, y_train_timepoint)
            y_pred_timepoint = model.predict(X_test_timepoint)

            if accuracy_metric == "accuracy":
                if model_name == 'RidgeClassifierCV':
                    accuracy_stored[t_i, g] = accuracy_score(y_test_timepoint, y_pred_timepoint)
                else:
                    accuracy_stored[t_i, g] = accuracy_score(y_test_timepoint, np.sign(y_pred_timepoint))
            elif accuracy_metric == "mean_squared_error":
                accuracy_stored[t_i, g] = mean_squared_error(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == "explained_variance":
                accuracy_stored[t_i, g] = explained_variance_score(y_test_timepoint, y_pred_timepoint)

        print("mean overall accuracy fold %d" % g)
        print(np.mean(accuracy_stored[:, g]))

    return accuracy_stored
# Define some useful functions

def run_timepoint_classification_cv(X, y, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy',
                                    return_coefficients=False, return_prediction=False):
    # we only use it for multiclass classification with ridge classifier!!!!
    # remember the y here is multiclass
    n_trials, n_units, n_time_points = X.shape
    indices = np.arange(n_trials)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    betas = np.zeros(shape=(len(np.unique(y)), n_units, n_time_points, n_folds))
    accuracy_stored = np.zeros(shape=(n_time_points, n_folds))
    prediction_logik = np.empty(shape=(n_time_points, n_trials))
    prediction_logik[:,:] = np.nan
    test_indices = []
    # loop on kfold splits
    k = 0
    y_pred_tot = np.zeros(shape=(n_trials, n_time_points))
    for train_index, test_index in kf.split(indices, y):
        # split everything into train and test
        # store test indices to use for gammas later
        test_indices.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train.shape)
        print(y_train.shape)

        # train on one time point
        for t_i in range(n_time_points):
            # create all training test split
            X_train_timepoint = X_train[:, :, t_i]
            X_test_timepoint = X_test[:, :, t_i]
            # make sure that you can handle y multidimension
            if len(y.shape) > 1:
                y_train_timepoint = y_train[:, t_i]
                y_test_timepoint = y_test[:, t_i]
            else:
                y_train_timepoint = y_train
                y_test_timepoint = y_test

            # the model has to be instantiated before every use
            if model_name == 'RidgeClassifierCV':
                model = RidgeClassifierCV()
            else:
                raise AssertionError("only RidgeClassifierCV available - to be used with multiclass target")

            model.fit(X_train_timepoint, y_train_timepoint)
            y_pred_timepoint = model.predict(X_test_timepoint)
            betas[:, :, t_i, k] = model.coef_
            prediction_logik[t_i, test_index] = (y_test_timepoint == y_pred_timepoint)*1.0
            if accuracy_metric == "accuracy":
                accuracy_stored[t_i, k] = accuracy_score(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == 'f1_score':
                accuracy_stored[t_i, k] = f1_score(y_test_timepoint, y_pred_timepoint, average='weighted')
            else:
                raise AssertionError('only f1 score available with multiclass task')

        print("mean overall accuracy fold %d" % k)
        print(np.mean(accuracy_stored[:, k]))
        k += 1

    if return_prediction:
        return (accuracy_stored, prediction_logik)
    elif return_coefficients:
        return (accuracy_stored, betas)
    else:
        return accuracy_stored


# Define some useful functions

def run_cv_timepoint_regression(X, y, model_name='RidgeCV', n_folds=10, return_coeff=True,
                                accuracy_metric='explained_variance'):
    n_trials, n_units, n_time_points = X.shape
    indices = np.arange(n_trials)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    accuracy_stored = np.zeros(shape=(n_time_points, n_folds))
    model_coefficients = np.zeros(shape=(n_units, n_time_points, n_folds))
    # loop on kfold splits
    k = 0
    y_pred_tot = np.zeros(shape=(n_trials, n_time_points))
    for train_index, test_index in kf.split(indices):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train.shape)
        print(y_train.shape)

        # train on one time point
        for t_i in range(n_time_points):

            # create all training test split
            X_train_timepoint = X_train[:, :, t_i]
            X_test_timepoint = X_test[:, :, t_i]
            # make sure that you can handle y multidimension
            if len(y.shape) > 1:
                y_train_timepoint = y_train[:, t_i]
                y_test_timepoint = y_test[:, t_i]
            else:
                y_train_timepoint = y_train
                y_test_timepoint = y_test

            # the model has to be instantiated before every use
            if model_name == 'RidgeClassifierCV':
                model = RidgeClassifierCV()
            elif model_name == 'RidgeCV':
                model = RidgeCV()
            elif model_name == 'ElasticNetCV':
                model = ElasticNetCV(max_iter=5000, tol=0.01)
            else:
                raise AssertionError("other models not available yet")

            model.fit(X_train_timepoint, y_train_timepoint)
            model_coefficients[:, t_i, k] = model.coef_
            y_pred_timepoint = model.predict(X_test_timepoint)
            y_pred_tot[test_index, t_i] = y_pred_timepoint

            if accuracy_metric == "accuracy":
                accuracy_stored[t_i, k] = accuracy_score(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == "mean_squared_error":
                accuracy_stored[t_i, k] = mean_squared_error(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == "explained_variance":
                accuracy_stored[t_i, k] = explained_variance_score(y_test_timepoint, y_pred_timepoint)
            elif accuracy_metric == 'f1_score':
                accuracy_stored[t_i, k] = f1_score(y_test_timepoint, y_pred_timepoint, average='weighted')

        print("mean overall accuracy fold %d" % k)
        print(np.mean(accuracy_stored[:, k]))
        k += 1

    if return_coeff:
        return model_coefficients, accuracy_stored
    else:
        return accuracy_stored


def run_simple_regresion(X, y, model_name='Ridge', accuracy_metric='explained_variance', statistical_testing=True,
                         n_permutations=1000):
    n_trials, n_states, n_time_points = X.shape
    prediction_accuracy = np.zeros(shape=(n_time_points,))
    prediction_accuracy_rand = np.zeros(shape=(n_permutations, n_time_points))
    trial_indices = np.arange(n_trials)

    for t_i in range(n_time_points):
        # trial test error to regress is not binary
        # (EV of the decoding of units, technically smaller than 1, could be negative if cross validated)

        y_timepoint = y[t_i, :]
        X_timepoint = X[:, :, t_i]

        # we need just ridge regression (no classifier).
        # no CV for now
        # the model has to be instantiated before every use
        if model_name == 'Ridge':
            model = Ridge()
        else:
            raise AssertionError("model not available yet")

        model.fit(X_timepoint, y_timepoint)

        y_pred_timepoint = model.predict(X_timepoint)
        if accuracy_metric == "classification":
            prediction_accuracy[t_i] = accuracy_score(y_timepoint, np.sign(y_pred_timepoint))
        elif accuracy_metric == "mean_squared_error":
            prediction_accuracy[t_i] = mean_squared_error(y_timepoint, y_pred_timepoint)
        elif accuracy_metric == "explained_variance":
            prediction_accuracy[t_i] = explained_variance_score(y_timepoint, y_pred_timepoint)

        if statistical_testing:
            # do permutation testing permuting the trials labels too
            for n in range(n_permutations):
                randomized_trials = trial_indices.copy()
                np.random.shuffle(randomized_trials)
                X_random = X_timepoint[randomized_trials]

                if model_name == 'Ridge':
                    model = Ridge()
                else:
                    raise AssertionError("model not available yet ")

                model.fit(X_random, y_timepoint)

                y_pred_timepoint_rand = model.predict(X_random)
                if accuracy_metric == "classification":
                    prediction_accuracy_rand[n, t_i] = accuracy_score(y_timepoint, np.sign(y_pred_timepoint_rand))
                elif accuracy_metric == "mean_squared_error":
                    prediction_accuracy_rand[n, t_i] = mean_squared_error(y_timepoint, y_pred_timepoint_rand)
                elif accuracy_metric == "explained_variance":
                    prediction_accuracy_rand[n, t_i] = explained_variance_score(y_timepoint, y_pred_timepoint_rand)
            # after the permutations compute the p value for this timepoint

    return (prediction_accuracy, prediction_accuracy_rand)


def compute_CI(x, confidence=0.95, axis=0):

    x_mean = np.mean(x, axis)
    CI = st.norm.interval(confidence=confidence,loc=x_mean,scale=st.sem(x, axis=axis))
    return x_mean, CI



# find how many data points belong to a specific target
def get_target_distribution(labels):
    indices = []
    classes = np.unique(labels)
    n_classes = len(classes)
    target_distribution = np.zeros(shape=(n_classes,))
    for i in range(n_classes):
        target_distribution[i] = np.sum(labels == classes[i])
        class_index = np.where(labels == classes[i])[0]
        indices.append(class_index)
    return (target_distribution, indices)


def get_trained_betas(X, y, model_name='RidgeCV', accuracy_metric='explained_variance', return_coeff=True):
    n_trials, n_units, n_time_points = X.shape
    model_coefficients = np.zeros(shape=(n_units, n_time_points))
    accuracy_stored = np.zeros(shape=(n_time_points,))
    # train on one time point
    for t_i in range(n_time_points):

        # create all training test split
        X_timepoint = X[:, :, t_i]

        # make sure that you can handle y multidimension
        if len(y.shape) > 1:
            y_timepoint = y[:, t_i]
        else:
            y_timepoint = y

        # the model has to be instantiated before every use
        if model_name == 'RidgeClassifierCV':
            model = RidgeClassifierCV()
        elif model_name == 'RidgeCV':
            model = RidgeCV()
        else:
            raise AssertionError("other models not available yet")

        model.fit(X_timepoint, y_timepoint)
        model_coefficients[:, t_i] = model.coef_
        y_pred_timepoint = model.predict(X_timepoint)

        if accuracy_metric == "accuracy":
            accuracy_stored[t_i] = accuracy_score(y_timepoint, np.sign(y_pred_timepoint))
        elif accuracy_metric == "mean_squared_error":
            accuracy_stored[t_i] = mean_squared_error(y_timepoint, y_pred_timepoint)
        elif accuracy_metric == "explained_variance":
            accuracy_stored[t_i] = explained_variance_score(y_timepoint, y_pred_timepoint)
        elif accuracy_metric == 'f1_score':
            accuracy_stored[t_i] = f1_score(y_timepoint, y_pred_timepoint, average='weighted')

    # print("mean overall accuracy best model")
    # print(np.mean(accuracy_stored))

    if return_coeff:
        return model_coefficients
    else:
        return accuracy_stored


def cross_corr_estimation(estimated_betas_state, lag):
    n_states, n_units, n_time_points, n_folds = estimated_betas_state.shape
    x = np.mean(estimated_betas_state[0, :, :, :], axis=2)
    y = np.mean(estimated_betas_state[1, :, :, :], axis=2)
    corr_lag = np.zeros(shape=(n_time_points - lag))
    # for each time point compute correlation with points
    for t_i in n_time_points - lag:
        t_j = t_i + lag
        xt = x[:, t_i]
        yt = y[:, t_j]
        corr_lag[t_i] = np.corrcoef(xt, yt)[0, 1]

    return corr_lag


def convert_vp_to_gamma(vp, states_change=None):
    n_trials,n_time_points = vp.shape
    if states_change is None:
        n_states = len(np.unique(vp))
        gamma_vp = np.zeros(shape=(n_trials, n_states, n_time_points))

        for t in range(n_time_points):
            for k in range(n_states):
                state_trials = np.where(vp[:,t] == k+1)[0]
                gamma_vp[state_trials, k, t] = 1

    elif states_change == 'merge': # merge 1 and 2
        n_states = len(np.unique(vp))
        gamma_vp = np.zeros(shape=(n_trials, n_states-1, n_time_points))

        for t in range(n_time_points):
            for k in range(n_states):
                state_trials = np.where(vp[:, t] == k + 1)[0]
                # manual hacking: merge state 1 and 2
                if k==0:
                    gamma_vp[state_trials, 0, t] = 1
                elif k==1:
                    gamma_vp[state_trials, 0, t] = 1
                else:
                    gamma_vp[state_trials, 1, t] = 1

    elif states_change == 'discard': # discard state 1
        n_states = len(np.unique(vp))
        gamma_vp = np.zeros(shape=(n_trials, n_states - 1, n_time_points))

        for t in range(n_time_points):
            for k in range(n_states):
                state_trials = np.where(vp[:, t] == k + 1)[0]
                # manual hacking: merge state 1 and 2
                if k == 0:
                    continue
                elif k == 1:
                    gamma_vp[state_trials, 0, t] = 1
                else:
                    gamma_vp[state_trials, 1, t] = 1

    else:
        raise AssertionError('only possibilities to change states are merge and discard')

    return gamma_vp


# code TGM for time/fixed y

def compute_standard_balanced_TGM(X, y, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy', return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # ------ establish the class imbalance -------
    nbr_class_1 = np.sum(y == 1)
    nbr_class_2 = np.sum(y == 0)

    if nbr_class_2 < nbr_class_1:
        smaller_class_nbr = nbr_class_2
        smaller_label = 0
        bigger_label = 1
    else:
        smaller_class_nbr = nbr_class_1
        smaller_label = 1
        bigger_label = 0

    # we will take 90% of these trials for training, and the same amount in the next
    trials_smallest_class = np.where(y == smaller_label)[0]  # smaller class trials indices
    trials_biggest_class = np.where(y == bigger_label)[0]  # bigger class trials indices

    # use kfolds only in the smallest batch
    training_nbr = int(0.8 * smaller_class_nbr)
    testing_nbr = int(0.2 * smaller_class_nbr)

    standard_TGM = np.zeros(shape=(n_time_points, n_time_points, n_folds))

    betas = np.zeros(shape=(n_units, n_time_points, n_folds))

    y_pred_tot = np.empty(shape=(n_trials, n_time_points, n_folds))
    y_pred_tot[:, :, :] = np.nan
    np.random.seed(10)
    for k in range(n_folds):
        # for every kfold shuffle the order of the trials, get first 90% as training, remaining 10% as testing,
        # same for the bigger class
        np.random.shuffle(trials_smallest_class)
        np.random.shuffle(trials_biggest_class)
        small_training_trials = trials_smallest_class[0:training_nbr]
        small_testing_trials = trials_smallest_class[training_nbr:training_nbr + testing_nbr]
        big_training_trials = trials_biggest_class[0:training_nbr]
        big_testing_trials = trials_biggest_class[training_nbr:training_nbr + testing_nbr]

        train_index = np.concatenate((small_training_trials, big_training_trials))
        np.random.shuffle(train_index)
        test_index = np.concatenate((small_testing_trials, big_testing_trials))
        np.random.shuffle(test_index)

        # split everything into train and test
        # store test indices to use for gammas later

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train on one time point
        for t_i in range(n_time_points):
            print('t = %d'%t_i)
            # create all training test split
            X_train_timepoint = X_train[:, :, t_i]
            y_train_timepoint = y_train

            # the model has to be instantiated before every use
            if model_name == 'RidgeClassifierCV':
                model = RidgeClassifierCV()
            elif model_name == 'ElasticNetCV':
                model = ElasticNetCV(max_iter=5000, tol=0.01)
            else:
                raise AssertionError("other models not available yet")

            model.fit(X_train_timepoint, y_train_timepoint)
            if return_coefficients:
                betas[:, t_i, k] = model.coef_

            # test on all time points
            for t_j in range(n_time_points):
                X_test_timepoint = X_test[:, :, t_j]
                y_test_timepoint = y_test

                y_pred_timepoint = model.predict(X_test_timepoint)

                if accuracy_metric == "accuracy":
                    if model_name == 'RidgeClassifierCV':
                        standard_TGM[t_j,t_i, k] = accuracy_score(y_test_timepoint, y_pred_timepoint)
                    else:
                        standard_TGM[t_j,t_i, k] = accuracy_score(y_test_timepoint, np.sign(y_pred_timepoint))

                elif accuracy_metric == "mean_squared_error":
                    standard_TGM[t_j,t_i, k] = mean_squared_error(y_test_timepoint, y_pred_timepoint)
                elif accuracy_metric == "explained_variance":
                    standard_TGM[t_j,t_i, k] = explained_variance_score(y_test_timepoint, y_pred_timepoint)

        k += 1

    if return_coefficients:
        return (standard_TGM, betas)
    else:
        return standard_TGM


# code TGM for time-varying y and state-wise data
"""
def compute_per_state_balanced_TGM(X, y, vp, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy', return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # y can be one dimensional or two dimensional, i.e. this can handle time varying y

    # if y is one dimensional, we assume it is the same for each time point and prolongue it
    if len(y.shape) == 1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

    # vp is expected to be the viterbi path, 3 dimensions
    n_states = vp.shape[1]
    TGM_per_state = np.zeros(shape = (n_states, n_time_points,n_time_points,n_folds))
    betas = np.zeros(shape = (n_states, n_units,n_time_points,n_folds))

    # first, permute trials
    trials_indices = np.arange(n_trials)
    state_trials_labels = {}
    for t in range(n_time_points):
        # list of trials belonging to each set per time point
        state_trials_labels[t] = sort_trials_by_variables(y[:, t], vp[:, 1, t])
        # spitted as:
        # 0A , 0B
        # 1A , 1B

    for p in range(n_folds):
        print('bootstrap %d'%p)
        # for this permutation, shuffle the trials within each category
        for t in range(n_time_points):
            np.random.shuffle(state_trials_labels[t][0][0])
            np.random.shuffle(state_trials_labels[t][0][1])
            np.random.shuffle(state_trials_labels[t][1][0])
            np.random.shuffle(state_trials_labels[t][1][1])

        # now for each trial
        for k in range(n_states):
            print('state %d'%k)
            for t_i in range(n_time_points):
                print('train point %d'%t_i)
                if len(state_trials_labels[t_i][0][k]) < 2 or len(state_trials_labels[t_i][1][k]) < 2:
                   # not enough trials for this state for this time point
                   #TGM_per_state[k,:,t_i,:] = np.nan
                   continue
                else:
                   # find minimum number of trials in this state to train
                   min_nbr_per_class = min(len(state_trials_labels[t_i][0][k]),len(state_trials_labels[t_i][1][k]))
                   train_nbr_per_class = int(0.8*min_nbr_per_class)

                   train_idx_class1 = np.array(state_trials_labels[t_i][0][k])[:train_nbr_per_class]
                   train_idx_class2 = np.array(state_trials_labels[t_i][1][k])[:train_nbr_per_class]

                   tot_train_trials = np.concatenate((train_idx_class1,train_idx_class2),0)
                   np.random.shuffle(tot_train_trials)

                   # train this state, this time point
                   model_ti = RidgeClassifierCV()
                   model_ti.fit(X[tot_train_trials,:,t_i],y[tot_train_trials,t_i])
                   betas[k,:,t_i,p] = model_ti.coef_

                   #test on each time point
                   for t_j in range(n_time_points):
                       # find minimum number of trials in this timepoint to test
                       min_nbr_per_class = min(len(state_trials_labels[t_j][0][k]), len(state_trials_labels[t_j][1][k]))
                       train_nbr_per_class = int(0.8 * min_nbr_per_class)
                       test_nbr_per_class = min_nbr_per_class - train_nbr_per_class

                       test_idx_class1 = np.array(state_trials_labels[t_j][0][k])[train_nbr_per_class:train_nbr_per_class+test_nbr_per_class]
                       test_idx_class2 = np.array(state_trials_labels[t_j][1][k])[train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                       tot_test_trials = np.concatenate((test_idx_class1,test_idx_class2),0)

                       y_pred_tj = model_ti.predict(X[tot_test_trials,:,t_j])
                       TGM_per_state[k,t_j,t_i,p] = accuracy_score(y[tot_test_trials,t_j],y_pred_tj)

    if return_coefficients:
        return (TGM_per_state, betas)
    else:
        return TGM_per_state
"""

def compute_state_aggregated_balanced_TGM(X, y, vp, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy',
                                   return_coefficients=False):
    # save predictions for both states and then compute accuracy aggregated across the whole set
    n_trials, n_units, n_time_points = X.shape
    # y can be one dimensional or two dimensional, i.e. this can handle time varying y

    # if y is one dimensional, we assume it is the same for each time point and prolongue it
    if len(y.shape) == 1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

    # vp is expected to be the viterbi path, 3 dimensions
    n_states = vp.shape[1]
    TGM_state_aggregated = np.zeros(shape=(n_time_points, n_time_points, n_folds))
    betas = np.zeros(shape=(n_states, n_units, n_time_points, n_folds))

    # first, permute trials
    trials_indices = np.arange(n_trials)
    for p in range(n_folds):
        # for each permutation
        # remember to save predictions and true labels
        print('bootstrap %d'%p)
        np.random.shuffle(trials_indices)
        # maybe no need for doing this - can shuffle only the trials needed
        Xp = X[trials_indices]
        yp = y[trials_indices]
        vpp = vp[trials_indices]

        state_trials_labels = {}

        # for this permutation, find all trials indices belonging to each state at each time point,
        for t in range(n_time_points):
            # list of trials belonging to each set per time point
            state_trials_labels[t] = sort_trials_by_variables(yp[:, t], vpp[:, 1, t])

            # spitted as:
            # 0A , 0B
            # 1A , 1B
        #train on one state, test on the other
        # training state

        for t_i in range(n_time_points):

            y_true_state = {}
            y_pred_state = {}
            # for each training point, train on one state , test on all test point
            for k in range(n_states):
                if len(state_trials_labels[t_i][0][k]) < 2 or len(state_trials_labels[t_i][1][k]) < 2:
                    # not enough trials for this state for this time point
                    # TGM_per_state[k,:,t_i,:] = np.nan
                    continue
                else:

                    y_true_state[k] = {}
                    y_pred_state[k] = {}
                    # find minimum number of trials in this state to train
                    min_nbr_per_class = min(len(state_trials_labels[t_i][0][k]), len(state_trials_labels[t_i][1][k]))
                    train_nbr_per_class = int(0.8 * min_nbr_per_class)

                    train_idx_class1 = np.array(state_trials_labels[t_i][0][k])[:train_nbr_per_class]
                    train_idx_class2 = np.array(state_trials_labels[t_i][1][k])[:train_nbr_per_class]

                    tot_train_trials = np.concatenate((train_idx_class1, train_idx_class2), 0)
                    np.random.shuffle(tot_train_trials)

                    # train this state, this time point
                    model_ti = RidgeClassifierCV()
                    model_ti.fit(Xp[tot_train_trials, :, t_i], yp[tot_train_trials, t_i])
                    betas[k, :, t_i, p] = model_ti.coef_

                    for t_j in range(n_time_points):

                        # find minimum number of trials in this timepoint to test
                        min_nbr_per_class = min(len(state_trials_labels[t_j][0][k]),
                                                len(state_trials_labels[t_j][1][k]))
                        train_nbr_per_class = int(0.8 * min_nbr_per_class)
                        test_nbr_per_class = min_nbr_per_class - train_nbr_per_class

                        test_idx_class1 = np.array(state_trials_labels[t_j][0][k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        test_idx_class2 = np.array(state_trials_labels[t_j][1][k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        tot_test_trials = np.concatenate((test_idx_class1, test_idx_class2), 0)

                        y_pred_tj = model_ti.predict(Xp[tot_test_trials, :, t_j])
                        y_true_state[k][t_j] = yp[tot_test_trials, t_j]
                        y_pred_state[k][t_j] = y_pred_tj

            for t_j in range(n_time_points):
                if 0 in y_pred_state.keys() and 1 in y_pred_state.keys():
                    time_point_pred = np.concatenate((y_pred_state[0][t_j],y_pred_state[1][t_j]), axis=0)
                    time_point_label = np.concatenate((y_true_state[0][t_j], y_true_state[1][t_j]), axis=0)
                elif 0 not in y_pred_state.keys():
                    time_point_pred = y_pred_state[1][t_j]
                    time_point_label = y_true_state[1][t_j]
                elif 1 not in y_pred_state.keys():
                    time_point_pred = y_pred_state[0][t_j]
                    time_point_label = y_true_state[0][t_j]
                # TGM[k] will spit the acc of a model trained on state k and tested on state 1-k
                TGM_state_aggregated[t_j, t_i, p] = accuracy_score(time_point_label, time_point_pred)

    if return_coefficients:
        return (TGM_state_aggregated, betas)
    else:
        return TGM_state_aggregated

def compute_per_state_balanced_TGM(X, y, vp, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy',
                                   return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # y can be one dimensional or two dimensional, i.e. this can handle time varying y

    # if y is one dimensional, we assume it is the same for each time point and prolongue it
    if len(y.shape) == 1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

    # vp is expected to be the viterbi path, 3 dimensions
    n_states = vp.shape[1]
    TGM_per_state = np.zeros(shape=(n_states, n_time_points, n_time_points, n_folds))
    betas = np.zeros(shape=(n_states, n_units, n_time_points, n_folds))

    # first, permute trials
    trials_indices = np.arange(n_trials)
    for p in range(n_folds):
        print('bootstrap %d'%p)
        np.random.shuffle(trials_indices)
        # maybe no need for doing this - can shuffle only the trials needed
        Xp = X[trials_indices]
        yp = y[trials_indices]
        vpp = vp[trials_indices]

        state_trials_labels = {}

        # for this permutation, find all trials indices belonging to each state at each time point,
        for t in range(n_time_points):
            # list of trials belonging to each set per time point
            state_trials_labels[t] = sort_trials_by_variables(yp[:, t], vpp[:, 1, t])

            # spitted as:
            # 0A , 0B
            # 1A , 1B
        #train on one state, test on the other
        # training state
        for k in range(n_states):
            print('state%d'%k)
            for t_i in range(n_time_points):
                if len(state_trials_labels[t_i][0][k]) < 2 or len(state_trials_labels[t_i][1][k]) < 2:
                    # not enough trials for this state for this time point
                    # TGM_per_state[k,:,t_i,:] = np.nan
                    continue
                else:
                    # find minimum number of trials in this state to train
                    min_nbr_per_class = min(len(state_trials_labels[t_i][0][k]), len(state_trials_labels[t_i][1][k]))
                    train_nbr_per_class = int(0.8 * min_nbr_per_class)

                    train_idx_class1 = np.array(state_trials_labels[t_i][0][k])[:train_nbr_per_class]
                    train_idx_class2 = np.array(state_trials_labels[t_i][1][k])[:train_nbr_per_class]

                    tot_train_trials = np.concatenate((train_idx_class1, train_idx_class2), 0)
                    np.random.shuffle(tot_train_trials)

                    # train this state, this time point
                    model_ti = RidgeClassifierCV()
                    model_ti.fit(Xp[tot_train_trials, :, t_i], yp[tot_train_trials, t_i])
                    betas[k, :, t_i, p] = model_ti.coef_

                    for t_j in range(n_time_points):
                        # find minimum number of trials in this timepoint to test
                        min_nbr_per_class = min(len(state_trials_labels[t_j][0][k]),
                                                len(state_trials_labels[t_j][1][k]))
                        train_nbr_per_class = int(0.8 * min_nbr_per_class)
                        test_nbr_per_class = min_nbr_per_class - train_nbr_per_class

                        test_idx_class1 = np.array(state_trials_labels[t_j][0][k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        test_idx_class2 = np.array(state_trials_labels[t_j][1][k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        tot_test_trials = np.concatenate((test_idx_class1, test_idx_class2), 0)

                        y_pred_tj = model_ti.predict(Xp[tot_test_trials, :, t_j])
                        # TGM[k] will spit the acc of a model trained on state k and tested on state 1-k
                        TGM_per_state[k, t_j, t_i, p] = accuracy_score(yp[tot_test_trials, t_j], y_pred_tj)

    if return_coefficients:
        return (TGM_per_state, betas)
    else:
        return TGM_per_state


def compute_cross_state_aggregated_balanced_TGM(X, y, vp, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy',
                                   return_coefficients=False):
    # save predictions for both states and then compute accuracy aggregated across the whole set
    n_trials, n_units, n_time_points = X.shape
    # y can be one dimensional or two dimensional, i.e. this can handle time varying y

    # if y is one dimensional, we assume it is the same for each time point and prolongue it
    if len(y.shape) == 1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

    # vp is expected to be the viterbi path, 3 dimensions
    n_states = vp.shape[1]
    TGM_cross_state_aggregated = np.zeros(shape=(n_time_points, n_time_points, n_folds))
    betas = np.zeros(shape=(n_states, n_units, n_time_points, n_folds))

    # first, permute trials
    trials_indices = np.arange(n_trials)
    for p in range(n_folds):
        # for each permutation
        # remember to save predictions and true labels
        print('bootstrap %d'%p)
        np.random.shuffle(trials_indices)
        # maybe no need for doing this - can shuffle only the trials needed
        Xp = X[trials_indices]
        yp = y[trials_indices]
        vpp = vp[trials_indices]

        state_trials_labels = {}

        # for this permutation, find all trials indices belonging to each state at each time point,
        for t in range(n_time_points):
            # list of trials belonging to each set per time point
            state_trials_labels[t] = sort_trials_by_variables(yp[:, t], vpp[:, 1, t])

            # spitted as:
            # 0A , 0B
            # 1A , 1B
        #train on one state, test on the other
        # training state

        for t_i in range(n_time_points):
            y_true_state = {}
            y_pred_state = {}
            # for each training point, train on one state , test on all test point
            for k in range(n_states):

                if len(state_trials_labels[t_i][0][k]) < 2 or len(state_trials_labels[t_i][1][k]) < 2:
                    # not enough trials for this state for this time point
                    # TGM_per_state[k,:,t_i,:] = np.nan
                    continue
                else:
                    y_true_state[k] = {}
                    y_pred_state[k] = {}

                    # find minimum number of trials in this state to train
                    min_nbr_per_class = min(len(state_trials_labels[t_i][0][k]), len(state_trials_labels[t_i][1][k]))
                    train_nbr_per_class = int(0.8 * min_nbr_per_class)

                    train_idx_class1 = np.array(state_trials_labels[t_i][0][k])[:train_nbr_per_class]
                    train_idx_class2 = np.array(state_trials_labels[t_i][1][k])[:train_nbr_per_class]

                    tot_train_trials = np.concatenate((train_idx_class1, train_idx_class2), 0)
                    np.random.shuffle(tot_train_trials)

                    # train this state, this time point
                    model_ti = RidgeClassifierCV()
                    model_ti.fit(Xp[tot_train_trials, :, t_i], yp[tot_train_trials, t_i])
                    betas[k, :, t_i, p] = model_ti.coef_

                    for t_j in range(n_time_points):

                        # find minimum number of trials in this timepoint to test
                        min_nbr_per_class = min(len(state_trials_labels[t_j][0][1-k]),
                                                len(state_trials_labels[t_j][1][1-k]))
                        train_nbr_per_class = int(0.8 * min_nbr_per_class)
                        test_nbr_per_class = min_nbr_per_class - train_nbr_per_class

                        test_idx_class1 = np.array(state_trials_labels[t_j][0][1-k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        test_idx_class2 = np.array(state_trials_labels[t_j][1][1-k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        tot_test_trials = np.concatenate((test_idx_class1, test_idx_class2), 0)

                        y_pred_tj = model_ti.predict(Xp[tot_test_trials, :, t_j])
                        y_true_state[k][t_j] = yp[tot_test_trials, t_j]
                        y_pred_state[k][t_j] = y_pred_tj

            # still at the level of training time point
            #compute aggregated prediction of the states
            for t_j in range(n_time_points):
                if 0 in y_pred_state.keys() and 1 in y_pred_state.keys():
                    time_point_pred = np.concatenate((y_pred_state[0][t_j],y_pred_state[1][t_j]), axis=0)
                    time_point_label = np.concatenate((y_true_state[0][t_j], y_true_state[1][t_j]), axis=0)
                elif 0 not in y_pred_state.keys():
                    time_point_pred = y_pred_state[1][t_j]
                    time_point_label = y_true_state[1][t_j]
                elif 1 not in y_pred_state.keys():
                    time_point_pred = y_pred_state[0][t_j]
                    time_point_label = y_true_state[0][t_j]

                TGM_cross_state_aggregated[t_j, t_i, p] = accuracy_score(time_point_label, time_point_pred)

    if return_coefficients:
        return (TGM_cross_state_aggregated,betas)
    else:
        return TGM_cross_state_aggregated


def compute_cross_state_balanced_TGM(X, y, vp, model_name='RidgeClassifierCV', n_folds=10, accuracy_metric='accuracy',
                                   return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # y can be one dimensional or two dimensional, i.e. this can handle time varying y

    # if y is one dimensional, we assume it is the same for each time point and prolongue it
    if len(y.shape) == 1:
        y = y[:, np.newaxis] * np.ones(shape=(n_trials, n_time_points))

    # vp is expected to be the viterbi path, 3 dimensions
    n_states = vp.shape[1]
    TGM_cross_state = np.zeros(shape=(n_states, n_time_points, n_time_points, n_folds))
    betas = np.zeros(shape=(n_states, n_units, n_time_points, n_folds))

    # first, permute trials
    trials_indices = np.arange(n_trials)
    for p in range(n_folds):
        np.random.shuffle(trials_indices)
        # maybe no need for doing this - can shuffle only the trials needed
        Xp = X[trials_indices]
        yp = y[trials_indices]
        vpp = vp[trials_indices]

        state_trials_labels = {}

        # for this permutation, find all trials indices belonging to each state at each time point,
        for t in range(n_time_points):
            # list of trials belonging to each set per time point
            state_trials_labels[t] = sort_trials_by_variables(yp[:, t], vpp[:, 1, t])

            # spitted as:
            # 0A , 0B
            # 1A , 1B
        #train on one state, test on the other
        # training state
        for k in range(n_states):
            for t_i in range(n_time_points):
                if len(state_trials_labels[t_i][0][k]) < 2 or len(state_trials_labels[t_i][1][k]) < 2:

                    # not enough trials for this state for this time point
                    # TGM_per_state[k,:,t_i,:] = np.nan
                    continue
                else:
                    # find minimum number of trials in this state to train
                    min_nbr_per_class = min(len(state_trials_labels[t_i][0][k]), len(state_trials_labels[t_i][1][k]))
                    train_nbr_per_class = int(0.8 * min_nbr_per_class)

                    train_idx_class1 = np.array(state_trials_labels[t_i][0][k])[:train_nbr_per_class]
                    train_idx_class2 = np.array(state_trials_labels[t_i][1][k])[:train_nbr_per_class]

                    tot_train_trials = np.concatenate((train_idx_class1, train_idx_class2), 0)
                    np.random.shuffle(tot_train_trials)

                    # train this state, this time point
                    model_ti = RidgeClassifierCV()
                    model_ti.fit(Xp[tot_train_trials, :, t_i], yp[tot_train_trials, t_i])
                    betas[k, :, t_i, p] = model_ti.coef_

                    # test on the other states, each time point
                    # THIS ASSUMES STATES ARE ONLY 2, with indices 1 and 0
                    # TODO: remember to make it more general if the states are more
                    for t_j in range(n_time_points):
                        # find minimum number of trials in this timepoint to test
                        min_nbr_per_class = min(len(state_trials_labels[t_j][0][1-k]),
                                                len(state_trials_labels[t_j][1][1-k]))
                        train_nbr_per_class = int(0.8 * min_nbr_per_class)
                        test_nbr_per_class = min_nbr_per_class - train_nbr_per_class

                        test_idx_class1 = np.array(state_trials_labels[t_j][0][1-k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        test_idx_class2 = np.array(state_trials_labels[t_j][1][1-k])[
                                          train_nbr_per_class:train_nbr_per_class + test_nbr_per_class]
                        tot_test_trials = np.concatenate((test_idx_class1, test_idx_class2), 0)

                        y_pred_tj = model_ti.predict(Xp[tot_test_trials, :, t_j])
                        # TGM[k] will spit the acc of a model trained on state k and tested on state 1-k
                        TGM_cross_state[k, t_j, t_i, p] = accuracy_score(yp[tot_test_trials, t_j], y_pred_tj)

    if return_coefficients:
        return (TGM_cross_state, betas)
    else:
        return TGM_cross_state




# ------------------------------------------- function for the glhmm ------------------
# here just for now
#TODO: fix the entropy across trials
import math
def get_Gamma_timepoint_entropy(Gamma_trials):
    """Calculates the entropy of each timepoint across a trial, if we understand fractional occupancies as probabilities.

    Parameters:
    --------------
    Gamma_trials : array-like of shape (n_trials, n_states, n_timepoints)
        The Gamma represents the state probability timeseries.
    Returns:
    --------
    entropy : array-like of shape (n_timepoints,)
        The entropy of each session.

    """

    N,K,T = Gamma_trials.shape
    entropy = np.zeros(shape=(T,))
    Gamma_avg = np.mean(Gamma_trials,axis=0)
    for t in range(T):
        for k in range(K):
            if Gamma_avg[k,t] == 0: continue
            entropy[t] -= math.log(Gamma_avg[k,t]) * Gamma_avg[k,t]
    return entropy


import math
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy import signal

def apply_ica(X,d,whitening=False,algorithm='parallel'):

    wpar = 'unit-variance' if whitening else False
    if d < 1:
        pcamodel = PCA(whiten=whitening)
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
    else:
        ncomp = d

    icamodel = FastICA(n_components=ncomp,whiten=wpar,algorithm=algorithm)
    icamodel.fit(X)
    X = icamodel.transform(X)

    # sign convention equal to Matlab's
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X

