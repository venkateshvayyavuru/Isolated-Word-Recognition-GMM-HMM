import numpy as np
import scipy.stats as st
import librosa as la
import numpy as np
import math
import soundfile as sf
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
identity = lambda x: x
class HMM(object):
 def __init__(self,num_states,num_comp):
  self.num_states=num_states
  self.random_state=np.random.RandomState(0)
  #Intilise the prior and transition probabilities
  self.priors=self.random_state.rand(self.num_states,1)
  self.priors=self._normalize(self.priors)
  self.A=self._stochasticize(self.random_state.rand(self.num_states, self.num_states))
  #Define the means and covariance of the state
  self.mu=None
  self.covs=None
  self.n_dim=None
  self.num_comp=num_comp
  self.lam=None
 def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
 def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
 #Forward Variable
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
 #Backward Variabel
 def backward(self, B):
  T = B.shape[1]
  beta = np.zeros(B.shape);
  beta[:, -1] = np.ones(B.shape[0])
  for t in range(T - 1)[::-1]:
   beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
   beta[:, t] /= np.sum(beta[:, t])
  return beta
 #Finding the state likelihoods
 def state_likelihood(self, observations):
   obs = observations
   B_comp=np.zeros((self.num_states,self.num_comp,obs.shape[1]))
   B = np.zeros((self.num_states, obs.shape[1]))
   for s in range(self.num_states):
     temp=np.zeros((obs.shape[1]))
     for c in range(self.num_comp):
      B_comp[s,c,:]=self.lam[s,c]*st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s, c].T, cov=self.covs[:, :, s, c].T)
      temp=temp+self.lam[s,c]*st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s, c].T, cov=self.covs[:, :, s, c].T)
     B[s, :]=temp
   return B,B_comp
 #Intialise the paramteres of the HMM in first iteration
 def _em_init(self, obs):
  if self.n_dim is None:
   self.n_dim = obs.shape[0]
  if self.num_comp==None:
   self.num_comp=3
  if self.lam is None:
   self.lam=np.zeros((self.num_states,self.num_comp))
   for i in range(self.num_comp):
    self.lam[:,i]=1/self.num_comp
  if self.mu is None:
   self.mu=np.zeros((self.n_dim,self.num_states,self.num_comp))
   for i in range(0,self.num_states):
    self.mu[:,i,:]=obs[:,i:i+self.num_comp]
  if self.covs is None:
   self.covs = np.zeros((self.n_dim, self.n_dim, self.num_states,self.num_comp))
   temp=np.zeros((self.n_dim, self.n_dim, self.num_states))
   temp +=np.diag(np.diag(np.cov(obs)))[:, :,None]
   for c in range(0,self.num_comp):
    for s in range(0,self.num_states):
     self.covs[:,:,s,c]=temp[:,:,s]
  return self


 #initilising the trellis for viterbi decoding
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
 #Baumwelch_resetimation 
 def baum_welch(self,obs):
  B,B_comp = self.state_likelihood(obs)
  #print("obeservation",B)
  #print("observations_comp",B_comp)
  T = obs.shape[1]
  log_like, alpha = self.forward(B)
  beta = self.backward(B)
  #ghama
  gamma_tjm=np.zeros((self.num_states,self.num_comp,obs.shape[1]))
  for t in range(obs.shape[1]):
   for j in range(self.num_states):
    for m in range(self.num_comp):
     a=alpha[j,t]*beta[j,t]/(sum(alpha[j,t]*beta[j,t] for j in range(self.num_states)))
     b=B_comp[j,m,t]/(sum(B_comp[j,m,t] for m in range(self.num_comp)))
     gamma_tjm[j,m,t]=a*b
  ghama_tj=np.sum(gamma_tjm,axis=1)
  #zeta
  zeta_tij=np.zeros((self.num_states,self.num_states,obs.shape[1]))
  expected_transitions=np.zeros((self.num_states,self.num_states))
  for i in range(self.num_states):
   for j in range(self.num_states):
    for t in range(obs.shape[1]-1):
     zeta_tij[i,j,t]=(alpha[i,t]*beta[j,t+1]*B[j,t+1])/sum(sum( alpha[i,t]*beta[j,t+1]*B[j,t+1] for j in range(self.num_states)) for i in range(self.num_states) )
    expected_transitions[i,j]=sum(zeta_tij[i,j,t] for t in range(0,obs.shape[1]-1))/sum(ghama_tj[i,t] for t in range(0,obs.shape[1]-1))
  expected_priors=ghama_tj[:,0]
  lam=np.zeros((self.num_states,self.num_comp))
  mu=np.zeros((self.n_dim,self.num_states,self.num_comp))
  covs=np.zeros((self.n_dim,self.n_dim,self.num_states,self.num_comp))
  #Parameters of gaussian
  for j in range(self.num_states):
   for m in range(self.num_comp):
    lam[j,m]=sum(gamma_tjm[j,m,t] for t in range(obs.shape[1]))/(sum(sum( gamma_tjm[j,m,t] for m in range(self.num_comp)) for t in range(obs.shape[1])))
    mu[:,j,m]=sum(gamma_tjm[j,m,t]*obs[:,t] for t in range(obs.shape[1]))/sum(gamma_tjm[j,m,t] for t in range(obs.shape[1]))
    covs[:,:,j,m]=sum(gamma_tjm[j,m,t]*np.outer((obs[:,t]-mu[:,j,m]),(obs[:,t]-mu[:,j,m]).T) for t in range(obs.shape[1]))/sum(gamma_tjm[j,m,t] for t in range(obs.shape[1]))
  #Ensure positive definite by loading diagonal elments
  for i in range(self.num_comp):
   covs[:,:,:,i] += .01 * np.eye(self.n_dim)[:, :, None]
  self.lam=lam
  self.mu=mu
  self.covs=covs
  self.A=expected_transitions
  self.priors=expected_priors
  return log_like,B
 def get_feats(self,string):
  x,fs=sf.read(string)
  x_feats=la.feature.mfcc(x, n_mfcc=13,sr=fs,n_fft=200,hop_length=80)
  x_feats=(x_feats-np.mean(x_feats,axis=0))/np.var(x_feats,axis=0)
  return x_feats
 def fit(self,wavs,string,iters):
  for iter in range(0,iters):
   for i in range(len(wavs)):
    x=self.get_feats(string+"/"+wavs[i])
    if i == 0 and iter == 0 :
     self._em_init(x)
    log_like,B = self.baum_welch(x)
   print("iteration:",iter,"log_like :",log_like)
  return self
#Fitting HMM_GMM For the Ones and Zeros
hmm1=HMM(5,3)  #Num_state num_comp
wav1=os.listdir("Ones")
hmm1.fit(wav1,"Ones",20) #20 iterations
#seq=hmm.viterbi_seq(feats[i],B)
wavzeros=os.listdir("Zeros")
hmm0=HMM(5,3) #Num_state num_comp
hmm0.fit(wavzeros,"Zeros",20) #20 iterations
g=open("Result_5_3comp_13dim_new.txt",'w')
#Classification
print("Testing")
#For one wavefiles
for i in range(len(wav1)):
  x=hmm0.get_feats("Ones/"+wav1[i])
  B0,B0_comp=hmm0.state_likelihood(x)
  log_like0,alpha=hmm0.forward(B0)
  B1,B1_comp=hmm1.state_likelihood(x)
  log_like1,alpha=hmm1.forward(B1)
  print(log_like0,log_like1)
  score=np.array(([log_like0,log_like1]))
  g.write("1"+"\t"+str(np.argmax(score))+"\n")
#For zero wavwfiles
for i in range(len(wavzeros)):
  x=hmm0.get_feats("Zeros/"+wavzeros[i])
  B0,B0_comp=hmm0.state_likelihood(x)
  log_like0,alpha=hmm0.forward(B0)
  B1,B1_comp=hmm1.state_likelihood(x)
  log_like1,alpha=hmm1.forward(B1)
  print(log_like0,log_like1)
  score=np.array(([log_like0,log_like1]))
  g.write("0"+"\t"+str(np.argmax(score))+"\n")
g.close()

