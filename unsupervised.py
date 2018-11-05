## MyDataPy
## Unsupervised methods shelf
## Simon Lebastard - Nov 2018

## REQUIREMENTS ##################################

### External requirements ########################
import importlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import math

### Internal requirements ########################
importlib.import_module('dataTools')

#################################################

## K-MEANS ######################################

class Kmeans:
  def __init__(self, data, nClass, ind=0, init='def'):
    # Create ID for instance + load data
    self.ind = ind
    self.data = data
    self.frame = np.zeros((2,2))
    self.frame[:,0] = self.data.min()
    self.frame[:,1] = self.data.max()
    
    # Check if data is non-empty, store data size
    [self.N,self.d] = np.shape(self.data)
    self.K = nClass
    
    # Initiate centroids
    self.centroids = np.zeros((self.K,self.d))
    initCentroids = np.zeros(self.K)
    if init == 'stoch':
      # Kmeans++ - computing probs for choosing initial centroids
      initCentroids[0] = np.random.randint(0,self.N-1)
      self.centroids[0,:] = self.data[np.array(initCentroids[0]).astype(int)]
      for index in range(self.K-1):
        distances = distance_matrix(self.data,self.centroids[:index+1,:])
        probs = np.amin(distances*distances,axis=1)
        probs[np.array(initCentroids).astype(int)] = 0
        Reg = np.sum(probs,axis=0)
        probs = probs/Reg
        initCentroids[index+1] = np.random.choice(self.N,1,p=probs)
        self.centroids[index+1,:] = self.data[np.array(initCentroids[index+1]).astype(int)]
      
    else:
      # Standard Kmeans - choose random data points as initial centroids
      initCentroids = np.random.choice(self.N, self.K, replace=False)
      self.centroids = self.data[np.array(initCentroids)]
    
    # Initiate assignment vector
    self.assignment = np.zeros(self.N)
  

  def assignCentroids(self):
    """Assigns to each data point the label of the closest centroid.
    Returns an array containing 1 for points that were changed assignment"""
    oldAssignment = self.assignment
    distances = distance_matrix(self.data,self.centroids)
    self.assignment = np.argmin(distances,axis=1)
    changeFlag = np.array_equal(oldAssignment,self.assignment)
    return changeFlag
  
    
  def computeCentroids(self):
    # Create indicator matrix - gives cluster belonging for each data point
    W = (self.assignment[:,None]==range(self.K)).astype(int)
    
    # Compute centroid
    for k in range(self.K):
      self.centroids[k,:] = np.average(self.data,axis=0,weights=W[:,k].reshape(self.N))

      
  def run(self):
    """Iterates centroid assignment and recomputation until convergence"""
    stop = False
    itCount = 0
    while not stop:
      stop = self.assignCentroids()
      self.computeCentroids()
      itCount += 1
    print('Convergence in {0:2d} iterations \n \n'.format(itCount))
 


  def draw(self,size=40,scale=0.8):
    listMarkers = ["s","d","*","+","h","x",'h','H','p','P','X']
    listColors = ['xkcd:purple','xkcd:green','xkcd:blue','xkcd:pink',
                  'xkcd:brown','xkcd:red','xkcd:light blue','xkcd:teal'
                  ,'xkcd:orange','xkcd:light green','xkcd:magenta',
                  'xkcd:yellow','xkcd:sky blue','xkcd:grey','xkcd:lime green',
                  'xkcd:light purple','xkcd:violet','xkcd:dark green',
                  'xkcd:turquoise','xkcd:lavender','xkcd:dark blue','xkcd:tan',
                  'xkcd:cyan','xkcd:aqua']
    plt.figure(self.ind, figsize=(15,15))
    for k in range(self.K):
      mark = np.random.choice(listMarkers,1)[0]
      color = np.random.choice(listColors,1)[0]
      plt.scatter(self.data[self.assignment==k][:,0], self.data[self.assignment==k][:,1], marker=mark, c=color, s=size, alpha=0.5)
      plt.scatter(self.centroids[k,0],self.centroids[k,1], marker=mark, c=color, s=3*size, alpha=1)
      axes = plt.gca()
      axes.set_xlim([scale*self.frame[0,0],(2-scale)*self.frame[0,1]])
      axes.set_ylim([scale*self.frame[1,0],(2-scale)*self.frame[1,1]])

    
    
  def stats(self):
    """Compute intra-cluster distortion, inter-cluster distortion, quality estimation for clustering"""

    W = (self.assignment[:,None]==range(self.K)).astype(int)
    distrt = np.zeros(self.K)
    relDistrt = np.zeros(self.K)
    effectives = np.zeros(self.K)

    interDist = distance_matrix(self.centroids,self.centroids)
    interDist[interDist==0] = np.nan
    globDistrt = np.nanmean(interDist*interDist)

    # Print general statistics
    print('Average inter-cluster distortion: {0:.2f} \n \n'.format(np.sqrt(globDistrt)))
    
    # Print statistics per cluster
    for k in range(self.K):
      effectives[k] = np.shape(self.data[W[:,k]==1])[0]
      distrt[k] = np.mean(np.power(np.linalg.norm(self.data[W[:,k]==1] - self.centroids[k,:],axis=1),2),axis=0)
      relDistrt[k] = distrt[k]/globDistrt
      
      effectives = effectives.astype(int)
      print('Cluster {0:3d}: {1:3d} entities'.format(k,effectives[k]))
      print('intra-cluster avg distortion {0:.2f}. Relative distortion {1:.2f}% \n'.format(np.sqrt(distrt[k]),100*np.sqrt(relDistrt[k])))
    
    print('Average relative distortion: {0:.2f}%'.format(np.average(100*np.sqrt(relDistrt),weights=effectives)))
    
    return [np.average(100*np.sqrt(relDistrt),weights=effectives),distrt,relDistrt]


################################################

## GAUSSIAN MIXTURE ###########################

class GaussianMixture:
  def __init__(self, K, d, pi, mu, sigma, isotropic = True):
    self.K, self.d = K, d
    self.isotropic = isotropic
    self.pi, self.mu, self.sigma = pi, mu, sigma
  
  def train(self, X, eps = 1e-4, max_iter = 10000, print_results = True):
    '''EM Algorithm for estimating gaussian mixture parameters'''
    log_likelihoods = []
    n, d = X.shape
    print("BEGIN EM ALGORITHM")
    
    while(len(log_likelihoods) < max_iter):
      #Expectation Step
      if self.isotropic:
        P = []
        for k in range(self.K):
          P.append(self.pi[k]*multivariate_normal.pdf(X, self.mu[k], sigma[k]).reshape(n, 1))
        P = np.hstack(P) #pi*Normal
      else:
        P = np.hstack([self.pi[k]*multivariate_normal.pdf(X, self.mu[k], self.sigma[k]).reshape(n, 1) for k in range(self.K)]) #pi*Normal
      Q = P/P.sum(axis=1)[:, np.newaxis]

      #Maximization step
      self.pi = Q.sum(axis=0)/Q.sum()
      self.mu = [X.T.dot(Q[:,k])/sum(Q[:, k]) for k in range(self.K)]
      if self.isotropic:
        for k in range(self.K):
          temp = (X - self.mu[k]).dot((X - self.mu[k]).T)
          Q_k = np.tile(Q[:, k], (n, 1)).T
          self.sigma[k] = np.trace(temp*Q_k)/(d*sum(Q[:, k]))*np.eye(d)
          
      else:
        self.sigma = [((X-self.mu[k]).T*Q[:,k]).dot(X-self.mu[k])/sum(Q[:, k]) for k in range(self.K)]

      #Log likelihood computation for exit condition
      log_likelihood = np.mean(np.log(np.sum(P, axis=1)))
      log_likelihoods.append(log_likelihood)
      if len(log_likelihoods) < 2:
        continue
      if np.abs(log_likelihoods[-2] - log_likelihood) < eps:
        print("Convergence Threshold reached")
        print("Last value of Average (Partial) Log_likelihood %0.2f" % log_likelihood)
        break
      
      
    if print_results:
      self.printResults(log_likelihoods)
      
      
  
  def predict(self, X):
    n = X.shape[0]
    
    P = np.hstack([self.pi[k]*multivariate_normal.pdf(X, self.mu[k], self.sigma[k]).reshape(n, 1) for k in range(0, self.K)])
    predictions = np.argmax(P, axis = 1)
    
    return predictions
  
  
  
  def printResults(self, log_likelihoods):
    '''Print the learnt parameters after training and the evolution of the partial
    log likelihood thourgh time.'''
    plt.plot(log_likelihoods)
    plt.ylabel('Average (Partial) Log Likelihood')
    plt.xlabel('Iteration')
    plt.title('Evolution of Average (Partial) Log Likelihood')
    plt.show()
    
    print("MU")
    for mu in self.mu: print(mu)
    print("\n")
    
    print("SIGMA - ISOTROPY = ", self.isotropic)
    for sigma in self.sigma: print(sigma)
    print("\n")
    
    print("PI")
    for pi in self.pi: print(pi)

      
      
  def print_log_likelihood(self, X):
    '''Print the average value of (partial) log_likelihood'''
    n = X.shape[0]
    
    if self.isotropic:
      P = []
      for k in range(self.K):
        P.append(self.pi[k]*multivariate_normal.pdf(X, self.mu[k], sigma[k]).reshape(n, 1))
      P = np.hstack(P)
    else:
      P = np.hstack([self.pi[k]*multivariate_normal.pdf(X, self.mu[k], self.sigma[k]).reshape(n, 1) for k in range(self.K)])
    
    print(np.mean(np.log(np.sum(P, axis=1))))
      
      
  def draw(self, data, predictions, size=40, scale=0.8, eps=0.1):
    """Prints data points, centroids and alineates the covariances matrices"""
    listMarkers = ["s","d","*","+","h","x",'h','H','p','P','X']
    listColors = ['xkcd:purple','xkcd:green','xkcd:blue','xkcd:pink',
                  'xkcd:brown','xkcd:red','xkcd:light blue','xkcd:teal'
                  ,'xkcd:orange','xkcd:light green','xkcd:magenta',
                  'xkcd:yellow','xkcd:sky blue','xkcd:grey','xkcd:lime green',
                  'xkcd:light purple','xkcd:violet','xkcd:dark green',
                  'xkcd:turquoise','xkcd:lavender','xkcd:dark blue','xkcd:tan',
                  'xkcd:cyan','xkcd:aqua']
    fig, ax = plt.subplots(figsize=(15,15))
    frame = np.zeros((2,2))
    frame[:,0] = data.min()
    frame[:,1] = data.max()
    for k in range(self.K):
      # Pick style
      mark = np.random.choice(listMarkers,1)[0]
      color = np.random.choice(listColors,1)[0]
      # Data points
      ax.scatter(data[predictions==k][:,0], data[predictions==k][:,1], marker=mark, c=color, s=size, alpha=0.5)
      # Centroid
      ax.scatter(self.mu[k][0],self.mu[k][1], marker=mark, c=color, s=3*size, alpha=1)
      # Covariance matrix delineation
      S = np.zeros((self.d,self.d))
      if self.isotropic:
        S = self.sigma[k]*np.identity(self.d)
      else:
        S = self.sigma[k]
      [E,V] = np.linalg.eig(S)
      #ang = np.angle(V[0,0]+V[0,1]*1j, deg=True)
      ang = np.degrees(np.arctan2(*V[:,0][::-1]))
      ell = Ellipse(self.mu[k],E[0]*np.sqrt(2*np.log(1/eps)),E[1]*np.sqrt(2*np.log(1/eps)),angle=ang,facecolor='none')
      ax.add_artist(ell)
      ell.set_clip_box(ax.bbox)
      ell.set_alpha(0.2)
      ell.set_facecolor(color)
    # Scaling
    axes = fig.gca()
    axes.set_xlim([scale*frame[0,0],(2-scale)*frame[0,1]])
    axes.set_ylim([scale*frame[1,0],(2-scale)*frame[1,1]])


###################################################