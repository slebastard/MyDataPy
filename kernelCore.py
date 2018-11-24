## MyDataPy
## Kernel core shelf
## Simon Lebastard - Nov 2018

## External requirements ###########################

## Data manipulation
import numpy as np

## Performance metrics
import time

## Kernel SVM requirements
from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False

from scipy.spatial.distance import cdist

## Importing self-made fcts
from metrics import *
from kernels import *

## Debugging
import pdb


## Internal requirements ##########################





## General class to build a kernel matrix
def build_kernel(arr1, arr2, kernel_fct, stringsData=True, verbose=True):
    """Builds the kernel matrix from numpy array @arr1 and @arr2 and kernel function @kernel_fct. V1, low-efficiency"""
    try:
        assert len(arr1) > 0
        assert len(arr2) > 0
    except AssertionError:
        print('At least one of the argument arrays is empty')
    if arr1.ndim == 1:
        arr1 = arr1.reshape((len(arr1),1))
    if arr2.ndim == 1:
        arr2 = arr2.reshape((len(arr2),1))
    
    if verbose:
        print('Building kernel matrix from {0:d}x{1:d} samples...'.format(len(arr1),len(arr2)))
    tick = time.time()
    
    if stringsData:
        K = cdist(arr1, arr2, lambda u, v: kernel_fct(u[0],v[0]))
    else:
        K = cdist(arr1, arr2, kernel_fct)
    
    if verbose:
        print('...done in {0:.2f}s'.format(time.time()-tick))
    return K


## 'Default' linear classifier (on numeric data only)
def linear_prod(x1, x2):
    """Returns the dot product between arrays @x1 and @x2"""
    t1 = np.ravel(x1)
    t2 = np.ravel(x2)
    if len(t1) != len(t2):
        raise ValueError("Undefined for sequences of unequal length")
    return np.dot(t1,t2)


## Variable assignement from kwargs
def get_from_KWargs(kwargs, name, default=None):
    """
    Extracts the value of attribute @name, from keyword attributes @kwargs.
    In case @name is not found in @kwargs, value @default is returned.
    """
    if name in kwargs:
        if kwargs[name] is not None:
            return kwargs[name]
    return default

## General method for k-fold cross validation
def kfold(data, labels, n_folds, train_method, pred_method, classify_method, labels_formatting, metric, target_folds, verbose=True, **kwargs):
    """
    Performs k-fold cross validation on your data, given a number of folds, a prediction and a classification method

    Parameters
    ----------
    data : N*p numpy array
    labels : N*1 numpy array
    n_folds : int
        number of folds for cross-validation. Must be strictly greater than 1
    train_method : str
        supervised method used for learning classifier. This is usually the learn function from a classifier object
    pred_method : str
        prediction method used for learning classifier. This is usually the predict function from a classifier object
    classify_method : str
        for classification problems, if pred_method outputs a continuous prediction as opposed to a discrete label prediction, this is the method you use for turning your continuous prediction into a label. This is usualy the classify function from a classifier object
    labels_formatting : str
        the method you use to format your classified predictions so that performance can be measured
    metric : str
        this is the performance metric you want to use to score the predictions of your algorithm
    target_folds : int
        you can choose to run cross-validation only on a subset of the data, by providing a set of target_folds to run on as a list
    verbose : boolean
        defaults True
    kwargs :
        those are the keyword arguments that the training and prediction methods use

    Returns
    -------
    float
        average of prediction scores across folds
    """
    try:
        assert n_folds > 1
    except AssertionError:
        print('Need more than one fold')

    try:
        assert len(data) == len(labels)
    except AssertionError:
        print('Error: Data and labels have different lengths')  
    
    if verbose: print('Engaging n-fold cross validation with {0:d} folds on {1:d} items'.format(n_folds, len(data)))    
    fold_size = int(len(data)/n_folds)
    # Random permuation of the data
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    res = []
    for fold in range(n_folds):
        if target_folds is not None and fold not in target_folds:
            res.append(np.nan)
            continue
        val_idx = range(fold*fold_size,(fold+1)*fold_size)
        val_data = np.array(data[val_idx])
        val_labels = np.array(labels[val_idx])

        train_data = np.array([element for i, element in enumerate(data) if i not in val_idx])
        train_labels = np.array([element for i, element in enumerate(labels) if i not in val_idx])

        train_method(train_data, train_labels, **kwargs)

        preds = pred_method(val_data, **kwargs)
        
        if metric.quantized:
            preds = classify_method(preds)
        res.append(metric.measure(np.ravel(preds), labels_formatting(val_labels)))
        if verbose: print('Fold {0:d}, {1:s}: {2:.2f}'.format(fold,metric.name,res[fold]))

    if verbose: print('Done! Average {0:s} is {1:.2f}'.format(metric.name,np.nanmean(res)))

    return np.nanmean(res)


###################################
### KERNEL METHODS PARENT CLASS ###
###################################

class kernelMethod():
    """
    Parent class for all kernel methods of the library
    """
    def __init__(self):
        pass

    def format_labels(self, labels):
        """Returns labels formatted for performance evaluation by a metric"""
        return labels

    def train(self, data, labels, kernel_fct=None, solver=None, stringsData=True, **kwargs):
        """
        Trains the classifier based on @data, @labels and a @kernel_fct.

        Parameters
        ::::::::-
        data : N*p numpy array
        labels : N*1 numpy array
        kernel_fct :
            optionnal, a method used to compute a kernel matrix from input data
        solver :
            optionnal, a numerical solver adapted to the task at hand
        stringsData : boolean
            indicating if we are dealing with strings
        kwargs :
            additional keyword arguments, for instance that should be provided to the solver or the kernel function
        
        Returns
        :::-

        """
        pass

    def predict(self, data, **kwargs):
        pass

    def classify(self, preds):
        return preds

    def assess(self, data, labels, n_folds=1, kernel_fct=None, solver=None, stringsData=True, metric=m_binary, target_folds=None, verbose=True):
        if n_folds > 1:
            return kfold(data, labels, n_folds, self.train, self.predict, self.classify, self.format_labels, metric, target_folds, verbose, format_labels=self.format_labels, stringsData=stringsData, kernel_fct=kernel_fct, solver=solver)    
        if n_folds == 1:
            self.train(data, labels, kernel_fct, solver, stringsData)
            preds = self.predict(data)
            if metric.quantized:
                preds = classify_method(preds)
            return metric.measure(np.ravel(preds), labels_formatting(val_labels))

    def grid_search(self, data, labels, hyperparameter, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        try:
            assert search_count > 1
            assert search_max > search_min
            assert folds_per_search > 0
        except AssertionError:
            print('One of arguments provided to grid-search is incorrect')

        grid = []
        total_folds = search_count*folds_per_search
        if n_folds is None:
            n_folds = total_folds

        for it in range(search_count):
            if scale == 'log':
                param = search_min*np.power(search_max*1.0/search_min,it*1.0/(search_count-1))
            else:
                param = search_min + it*1.0/(search_count-1)

            t_folds = np.remainder(range(it*folds_per_search,(it+1)*folds_per_search),n_folds-1)
            setattr(self, hyperparameter, param)
            grid.append({'value':param, 'folds':t_folds ,'score':self.assess(data, labels, n_folds, kernel_fct, solver=None, stringsData=False, metric=m_binary, target_folds=t_folds, verbose=False)})

        return grid