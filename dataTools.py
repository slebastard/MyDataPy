## MyDataPy
## Data processing shelf
## Simon Lebastard - Nov 2018

## External requirements ###########################

import numpy as np
import pandas as pd

# Identification with Google account to access data
#from google.colab import auth
#auth.authenticate_user()
#
## This shelf requires gspread. To install:
## !pip install --upgrade -q gspread
#import gspread
#from oauth2client.client import GoogleCredentials
#
#gc = gspread.authorize(GoogleCredentials.get_application_default())

## Internal requirements ##########################


###################################################

###############################
# Data loading and formatting #
###############################

def load_data(dsID, set_type='tr', folder_name='data'):
	"""Loads a dataset from a folder name and a dataset number

    Keyword arguments:
    dsID -- the dataset number. Your input data should be stored in files that look like 'Xk.csv', where k=dsID
    set_type -- the imaginary part (default 0.0)
    folder_name -- folder where your data is stored

    Returns: pandas dataframe containing data with index starting from 0

    ToDo: allow for this function to take as input any file name, with a defaut convention name
    """
    Xdata_file = folder_name + '/X' + set_type + str(dsID) + '.csv'
    X = pd.read_csv(Xdata_file, header=None, names=['Sequence'], dtype={'Sequence': np.unicode_})
    if set_type=='tr':
        Ydata_file = folder_name + '/Y' + set_type + str(dsID) + '.csv'
        Y = pd.read_csv(Ydata_file, index_col=0, dtype={'Bound': np.dtype(bool)})
        Y.index = Y.index - 1000*dsID
        df = pd.concat([X, Y], axis=1)
    else:
        df = X
    return df


def format_preds(preds):
	""" Translates signed predictions (-1/1 or signed with amplitude for confidence) into 0/1 predictions"""
    return (0.5*(1+np.sign(preds))).astype(int)


def data_normalization(data, offset_column=False):
	"""Performs data normalization

    Keyword arguments:
    data -- numpy array
    offset_column -- True if you want a column of ones appended at the bottom of your data

    Returns: pandas dataframe normalized, and optionally offset
    """
    d_mean = np.mean(data, axis=0)
    d_std = np.std(data, axis=0)
    data = (data - d_mean)/d_std
    if offset_column:
        data = np.hstack((data,np.ones((len(data),1))))
    return data


#####################################
# Weighting different classifiers   #
# to potentially do better than all #
#####################################

def voting(preds, wghts, stochastic=False):
	"""Produces a label prediction from many predictors

    Keyword arguments:
    preds -- Array of predictors. Each predictor is an array of predictions, of a given size N
    wghts -- Confidence Weights given to the respective predictors
	stochastic -- If you set this to be True, the consensus prediction will be chosen from a binomial distribution from the different prediction votes

    Returns: array of N label predictions
    """	
    votes =  np.average(preds, axis=1, weights=wghts)
    if stochastic:
    	return np.random.binomial(1, p=votes).astype(int)
    else:
    	return (0.5*(1 + np.sign(votes-0.5))).astype(int)


##########################################
# Mutual-information based dim reduction #
##########################################

def get_MI(data, labels, word_idx, bins):
	"""Returns the mutual information between a word and a binary label

    Keyword arguments:
    data -- numpy array
    labels -- numpy array of booleans
	word_idx -- the index corresponding to the word you wish to compute MI for. You must have defined a table mapping word_idxs to words before you can use this function
	bins -- Discretization bins for probability computation

    Returns: mutual information between word and binary label
    """		
    n,p = data.shape
    idx_bound = np.argwhere(labels==1)
    idx_unbound = np.argwhere(labels==0)
    data_bound = np.take(data, idx_bound, axis=0)
    data_unbound = np.take(data, idx_unbound, axis=0)
    
    n_b = len(data_bound)
    n_ub = n - n_b
    data_bound = data_bound.reshape((n_b,p))
    data_unbound = data_unbound.reshape((n_ub,p))
    
    p_b = n_b*1.0/n
    p_ub = 1.0 - p_b
    
    MI = 0
    for abin in bins:
        b_cond = np.count_nonzero(np.isin(data_bound[:,word_idx], abin))*1.0/n_b
        ub_cond= np.count_nonzero(np.isin(data_unbound[:,word_idx], abin))*1.0/n_ub

        cond_data = np.isin(data[:,word_idx], abin)
        n_cond = np.count_nonzero(cond_data)
        if n_cond == 0:
            continue
        cond_b = np.count_nonzero(labels[cond_data]==1)*1.0/n_cond
        cond_ub = 1.0 - cond_b       
    
        if cond_b > 0:
            MI = MI + b_cond*p_b*np.log(cond_b/p_b)
        if cond_ub > 0:
            MI = MI + ub_cond*p_ub*np.log(cond_ub/p_ub)
        if np.isnan(MI):
            pdb.set_trace()
    return MI

def argmax_MI(data, labels, n_feats, bins):
	"""Returns the n_feats words that share the most information with the binary label

    Keyword arguments:
    data -- numpy array
    labels -- numpy array of booleans
	n_feats -- number of high-information words to yield
	bins -- Discretization bins for probability computation

    Returns: mutual information between word and binary label
    """	
    n,p = data.shape
    MI = np.zeros(p)
    for word_idx in range(p):
        MI[word_idx] = get_MI(data, labels, word_idx, bins)
    max_MI_idx = np.argsort(MI)[-1:-(n_feats+2):-1]
    return max_MI_idx, MI[max_MI_idx]

def MI_dimRed(data, labels, n_feats, bins):
	"""Reduces the dimensionality of a bag-of-words representation based on mutual information

    Keyword arguments:
    data -- numpy array
    labels -- numpy array of booleans
	n_feats -- number of high-information words to yield
	bins -- Discretization bins for probability computation

    Returns: N*n_feats numpy array, reduced BoW representation
    """	
    idx, MI_ranked = argmax_MI(data, features, n_feats, bins)
    data_lowdim = np.take(data, idx, axis=1)
    return data_lowdim, idx, MI_ranked