import numpy
import numpy as np
import scipy.stats as st
import librosa as la
import numpy as np
import math
import soundfile as sf
import matplotlib.pyplot as plt
import os
identity = lambda x: x
class HMM(object):
 def __init__(self,num_states):
  self.num_states=num_states
  self.start_state=0
  self.end_state=num_states-1
  self.random_state=np.random.RandomState(0)
  #Intilise the prior and transition probabilities
  self.priors=self.random_state.rand(self.num_states,1)
  self.priors[0]=0
  self.priors[-1]=0
  self.priors=None
  self.A=None
  #Define the means and covariance of the state
  self.mu=None
  self.covs=None
  self.n_dim=None
  self.num_comp=None
  self.lam=None
 def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
 def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
 def forward(self,B): #Finding the log-likelihood
  log_like=0
  T=B.shape[1] #Seq_length
  alpha=np.zeros((self.num_states,T))
  for t in range(T):
   if t==0 : #Intialisation
     alpha[:,t]=B[:,t]*self.priors.ravel() ##Vectorisation
   else:
     alpha[:,t]=B[:,t]*np.dot(self.A.T,alpha[:,t-1]) #Vectorisation
   alpha_sum=np.sum(alpha[:,t])
   alpha[:, t] /= alpha_sum
   log_like=log_like + np.log(alpha_sum)
  return log_like, alpha
 def backward(self, B):
  T = B.shape[1]
  beta = np.zeros(B.shape);
  beta[:, -1] = np.ones(B.shape[0])
  for t in range(T - 1)[::-1]:
   beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
   beta[:, t] /= np.sum(beta[:, t])
  return beta
 def state_likelihood(self, observations):
   obs = observations
   B_comp=np.zeros((self.num_states,self.num_comp,obs.shape[1]))
   B = np.zeros((self.num_states, obs.shape[1]))
   for s in range(1,self.num_states-1):
     temp=np.zeros((obs.shape[1]))
     for c in range(self.num_comp):
      B_comp[s,c,:]=self.lam[s,c]*st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s, c].T, cov=self.covs[:, :, s, c].T)
      temp=temp+self.lam[s,c]*st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s, c].T, cov=self.covs[:, :, s, c].T)
     B[s, :]=temp
   return B
 def _init_trellies(self,observations,B,init_func=identity):
  trellis = [ [None for j in range(observations.shape[1])]
                          for i in range(self.num_states)]
  v = lambda s: B[s,0]*self.priors[s]
  for state in range(self.num_states):
   trellis[state][0] = init_func(v(state))
  return trellis
 def back_trace(self,trellis,start):
  point=start[0]
  seq=[point,self.num_states-1]
  for t in reversed(range(0, len(trellis[1]))):
   val, backs = trellis[point][t]
   point = backs[0]
   seq.insert(0, point)
  return seq
 ##Finding the vitebi path
 def viterbi_seq(self,obs,B):
  trellis=self._init_trellies(obs,B,init_func=lambda val:(val,[0]))
  for t in range(1,obs.shape[1]):
   for state in range(0,self.num_states):
    prev=[(old_state,trellis[old_state][t-1][0]*self.A[old_state,state]*B[state,t]) for old_state in range(0,self.num_states)]
    high_value=max(prev, key=lambda p :p[1])[1]
    backs = [s for s, val in prev if val == high_value]
    trellis[state][t]=(high_value,backs)
  prev =[(old_state, trellis[old_state][-1][0] *self.A[old_state,(self.num_states)-1]) for old_state in range(0,self.num_states)]
  high_value=max(prev,key=lambda p :p[1])[1]
  backs=[s for s ,value in prev if value == high_value]
  seq = self.back_trace(trellis, backs)
  return seq
 def _em_init(self, obs):
  self.priors=np.array(([0,0.2,0.3,0.5,0]))
  self.A=np.array(([[0.0,0.5,0.4,0.1,0.0],[0.0,0.7,0.1,0.1,0.1],[0.0,0.1,0.7,0.1,0.1],[0.0,0.1,0.5,0.1,0.3],[0.0,0.0,0.0,0.0,1.0]]))
  if self.n_dim is None:
   self.n_dim = obs.shape[0]
  self.num_comp=3
  if self.lam is None:
   self.lam=np.zeros((self.num_states,self.num_comp))
   self.lam[1:-1,0]=0.3
   self.lam[1:-1,1]=0.3
   self.lam[1:-1,2]=0.4
  if self.mu is None:
   self.mu=np.zeros((self.n_dim,self.num_states,self.num_comp))
   self.mu[:,1,:]=obs[:,10:13]
   self.mu[:,2,:]=obs[:,15:18]
   self.mu[:,3,:]=obs[:,1:4]
  if self.covs is None:
   self.covs = np.zeros((self.n_dim, self.n_dim, self.num_states,3))
   temp=np.zeros((self.n_dim, self.n_dim, self.num_states))
   temp +=np.diag(np.diag(np.cov(obs)))[:, :,None]
   for c in range(0,self.num_comp):
    for s in range(1,self.num_states-1):
     self.covs[:,:,s,c]=temp[:,:,s]
  return self
hmm=HMM(5)
x,fs=sf.read("0_jackson_1.wav")
x_feats=la.feature.mfcc(x, n_mfcc=13,sr=fs,n_fft=200,hop_length=80)
x_feats=(x_feats-np.mean(x_feats,axis=0))/np.var(x_feats,axis=0)
hmm._em_init(x_feats)
B=hmm.state_likelihood(x_feats)
#Forward
log_like,alpha=hmm.forward(B)
#Backward
beta=hmm.backward(B)
#Viterbi
best_seq=hmm.viterbi_seq(x_feats,B)

