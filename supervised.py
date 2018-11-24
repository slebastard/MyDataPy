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
from kernelCore import *

## Debugging
import pdb


##################
### KERNEL SVM ###
##################

class kernelSVM(kernelMethod):
    """
    SVM instance, allowing for any kernel
    """
    def __init__(self, lbda=0.1, solver='cvxopt'):
        self.lbda = lbda
        self.solver = solver
        self.data = None
        self.alpha = None
        self.kernel_fct = None
    
    def format_labels(self, labels):
        """
        Transform any binary system of labels into the +1/-1 equivalent system
        """
        try:
            assert len(np.unique(labels)) == 2
        except AssertionError:
            print('Error: Labels provided are not binary')
        lm,lM = np.min(labels), np.max(labels)
        l = (labels==lM).astype(int) - (labels==lm).astype(int)
        return l
    
    def train(self, data, labels, **kwargs):
        """
        Trains the kernel SVM on data and labels
        """
        # Default kernel will be linear (only works in for finite-dim floats space)
        kernel_fct = get_from_KWargs(kwargs, 'kernel_fct', linear_prod)
        stringsData = get_from_KWargs(kwargs, 'stringsData', True)
        solver = get_from_KWargs(kwargs, 'solver', 'cvxopt')
        reg = get_from_KWargs(kwargs, 'reg', 0)
        verbose = get_from_KWargs(kwargs, 'verbose', False)

        n_samples = labels.shape[0]
        # Turning labels into ±1
        labels = self.format_labels(labels)
        # Binding kernel fct and data as attribute for further predictions
        self.kernel_fct = kernel_fct
        self.data = data
        # Building matrices for solving dual problem
        K = build_kernel(data, data, kernel_fct, stringsData, verbose)
        d = np.diag(labels)
        P = matrix(2.0*K + reg*np.eye(n_samples), tc='d')
        q = matrix(-2.0*labels, tc='d')
        G = matrix(np.vstack((-d,d)), tc='d')
        h1 = np.zeros((n_samples,1))
        h2 = (1.0/(2*self.lbda*n_samples))*np.ones((n_samples,1))
        h = matrix(np.vstack((h1,h2)), tc='d')
        # Construct the QP, invoke solver
        sol = solvers.qp(P,q,G,h,solver=solver)
        # Extract optimal value and solution
        self.alpha = np.asarray(sol['x'])
   
    def predict(self, data, **kwargs):
        """Predict labels for data"""
        try:
            assert self.alpha is not None
            assert self.kernel_fct is not None
        except AssertionError:
            print('Error: No successful training recorded')

        stringsData = get_from_KWargs(kwargs, 'stringsData', True)
        verbose = get_from_KWargs(kwargs, 'verbose', False)

        # Build sv alpha and sv K(x_i(new_data), x_j(ref))
        sv_ind = np.nonzero(self.alpha)[0]
        sv_alpha = self.alpha[sv_ind]
        sv_K = build_kernel(data, self.data[sv_ind], self.kernel_fct, stringsData, verbose)
        # Use supvec alpha and supvec K to compute predictions
        return sv_K @ sv_alpha

    def classify(self, preds):
        return np.sign(preds).astype(int)

    def grid_search(self, data, labels, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        return super().grid_search(data, labels, 'lbda', search_min, search_max, search_count, n_folds, scale, folds_per_search, kernel_fct)


##################
### KERNEL kNN ###
##################

class kernelKNN(kernelMethod):
    """K-nearest neighbor instance, allowing for any kernel"""
    def __init__(self, k):
        self.k = k

    def train(self, data, labels, **kwargs):
        self.ref_data = data
        self.ref_labels = labels

    def predict(self, data, **kwargs):
        ##  first let's find kNN for all points in the dataset
        kernel_fct = get_from_KWargs(kwargs, 'kernel_fct', linear_prod)

        self.kernel_fct = kernel_fct
        K = build_kernel(data, self.ref_data, self.kernel_fct, stringsData=False)
        idx = (np.argsort(K)[:,-self.k:])
        labels = np.array(self.ref_labels)[idx]
        bincount = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=labels)
        return np.argmax(bincount, axis=1)

    def grid_search(self, data, labels, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        return super().grid_search(data, labels, 'k', search_min, search_max, search_count, n_folds, scale, folds_per_search, kernel_fct)
        
        
##################################
### KERNEL LOGISTIC REGRESSION ###
##################################
def sigmoid(x):
    z = 1.0/(1.0 + np.exp(-x))
    return z

class kernelLogisticRegression(kernelMethod):
    """Logistic regression instance, allowing for any kernel"""
    def __init__(self, lbda=0.1):
        self.lbda = lbda
        self.data = None
        self.alpha = None
        self.kernel_fct = None
    
    def format_labels(self, labels):
        try:
            assert len(np.unique(labels)) == 2
        except AssertionError:
            print('Error: Labels provided are not binary')
        lm,lM = np.min(labels), np.max(labels)
        l = (labels==lM).astype(int) - (labels==lm).astype(int)
        return l
    
    def train(self, data, labels, max_iter = 10000, cvg_threshold = 1e-4, **kwargs):
        """Trains the kernel Logistic Regression on data and labels"""
        # Default kernel will be linear (only works in for finite-dim floats space)
        kernel_fct = get_from_KWargs(kwargs, 'kernel_fct', linear_prod)
        stringsData = get_from_KWargs(kwargs, 'stringsData', True)
        reg = get_from_KWargs(kwargs, 'reg', 0)
        
        self.max_iter = max_iter
        self.cvg_threshold = cvg_threshold
        
        n_samples = labels.shape[0]
        # Turning labels into ±1
        labels = self.format_labels(labels)
        
        # Binding kernel fct and data as attribute for further predictions
        self.kernel_fct = kernel_fct
        self.data = data
        
        # Building matrices for solving dual problem
        K = build_kernel(data, data, kernel_fct, stringsData)
        
        #Initialization
        alpha_t = 0.001*np.random.normal(0,1,n_samples)

        for i in range(self.max_iter):
            m_t = np.dot(K, alpha_t) #Shape: n x 1
            P_t = -sigmoid(-labels*m_t) #Shape n x 1
            W_t = sigmoid(m_t)*sigmoid(-m_t) #Shape n x 1
            z_t = m_t + labels/sigmoid(-labels*m_t) #Shape n x 1

            #Solve WKRR
            W_sqrt_matrix = np.diag(np.sqrt(W_t)) #Shape n x n
            temp = W_sqrt_matrix.dot(K.dot(W_sqrt_matrix)) + n_samples*self.lbda*np.eye(n_samples)
            temp = np.linalg.inv(temp) #Shape n x n
            alpha_next = np.dot(W_sqrt_matrix.dot(temp.dot(W_sqrt_matrix)), labels) #Shape n x 1

            if np.linalg.norm(alpha_next - alpha_t) < np.linalg.norm(alpha_t)*self.cvg_threshold:
                #print('Cvg achieved after {} iterations'.format(i))
                break
            else:
                alpha_t = alpha_next
        
        self.alpha = alpha_t
   
    
    def classify(self, preds):
        return np.sign(preds).astype(int)

    def predict(self, data, **kwargs):
        """Predict labels for data"""
        try:
            assert self.alpha is not None
            assert self.kernel_fct is not None
        except AssertionError:
            print('Error: No successful training recorded')

        stringsData = get_from_KWargs(kwargs, 'stringsData', True)

        # Build sv alpha and sv K(x_i(new_data), x_j(ref))
        sv_ind = np.nonzero(self.alpha)[0]
        sv_alpha = self.alpha[sv_ind]
        sv_K = build_kernel(data, self.data[sv_ind], self.kernel_fct, stringsData)
        # Use supvec alpha and supvec K to compute predictions
        return sv_K @ sv_alpha