import numpy as np
import math
import librosa as la
import os
import soundfile as sf
import matplotlib.pyplot as plt
#Distance matrix using cosine or ecludian
#If cosine True disnace in similarity
def distance_matrix_fun(x,y,metric_type):
 distance_matrix=np.zeros((len(x),len(y)))
 for i in range(0,len(x)):
  for j in range(0,len(y)):
   if metric_type == "cosine" :
     a=np.linalg.norm(x[i])
     b=np.linalg.norm(y[j])
     dot=np.dot(x[i],y[j])
     distance_matrix[i,j]=dot/(a*b)
   elif metric_type == "ecludian" :
    distance_matrix[i,j]=sum((x[i][k]-y[j][k])**2 for k in range(0,x.shape[1]))
 return distance_matrix
#Boundary conditions
def accumulated_mat(distance_matrix,type_cost):
 for i in range(1,accumulated_cost_matrix.shape[0]):
  for j in range(1,accumulated_cost_matrix.shape[1]):
   if i ==1 or j==1 :
    accumulated_cost_matrix[i,j]=distance_matrix[i,j]
   else:
    if type_cost=="cosine":
     accumulated_cost_matrix[i,j]=distance_matrix[i,j]+max(accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i-1,j-1])
    elif type_cost =="ecludian":
     accumulated_cost_matrix[i,j]=distance_matrix[i,j]+min(accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i-1,j-1])
 return accumulated_cost_matrix

#back tracking for best path

def backtrack(accumulated_cost_matrix,type_cost):
 path=[]
 path.append([accumulated_cost_matrix.shape[0]-1,accumulated_cost_matrix.shape[1]-1])
 i=accumulated_cost_matrix.shape[0]-1
 j=accumulated_cost_matrix.shape[1]-1
 while i>1 or j >1:
   if type_cost == "cosine":
    if accumulated_cost_matrix[i,j-1]==max(accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j-1]):
     j=j-1
    elif accumulated_cost_matrix[i-1,j]==max(accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j-1]):
     i=i-1
    else:
     i=i-1
     j=j-1
   elif type_cost =="ecludian":
    if accumulated_cost_matrix[i,j-1]==min(accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j-1]):
     j=j-1
    elif accumulated_cost_matrix[i-1,j]==min(accumulated_cost_matrix[i-1,j],accumulated_cost_matrix[i,j-1],accumulated_cost_matrix[i-1,j-1]):
     i=i-1
    else:
     i=i-1
     j=j-1
   path.append([i,j])
 return path
#Distance Plot function
def plot_distance_cost(distances): 
 im = plt.imshow(distances, interpolation='nearest', cmap='OrRd')  
 plt.gca().invert_yaxis() 
 plt.xlabel("X") 
 plt.ylabel("Y") 
 plt.grid() 
 plt.colorbar();
#plot_distance_cost(accumulated_cost_matrix)
#plt.show() 
g=open("cost_0_jackson_1_ref_Zeros_cosine.txt",'w')  
ref,fs=sf.read("0_jackson_1.wav")
ref_feats=la.feature.mfcc(ref, n_mfcc=10,sr=fs,n_fft=200,hop_length=80).T
ref_feats=(ref_feats-np.mean(ref_feats,axis=0))/np.var(ref_feats,axis=0)
list_wavs=os.listdir("Zeros")
for wav in list_wavs:
 y,f2=sf.read("Zeros/"+wav)
 y_feats=la.feature.mfcc(y, n_mfcc=10,sr=f2,n_fft=200,hop_length=80).T
 y_feats=(y_feats-np.mean(y_feats,axis=0))/np.var(y_feats,axis=0)
 distance_matrix1=distance_matrix_fun(ref_feats,y_feats,"cosine")
 distance_matrix=np.zeros((len(ref_feats)+1,len(y_feats)+1))
 distance_matrix[0,1:]=math.inf
 distance_matrix[1:,0]=math.inf
 distance_matrix[1:,1:]=distance_matrix1
#Acucumulated distaces matrix
 accumulated_cost_matrix=np.zeros_like(distance_matrix)
 accumulated_cost_matrix[0,1:]=math.inf
 accumulated_cost_matrix[1:,0]=math.inf
 accumulated_cost_matrix=accumulated_mat(distance_matrix,"cosine")
 path=backtrack(accumulated_cost_matrix,"cosine")
 cost=0
 for x ,y in path:
  cost=cost+sum((ref_feats[x-1][k]-y_feats[y-1][k])**2 for k in range(0,ref_feats.shape[1]))
 g.write(wav+"\t"+str(cost/len(path))+"\n")
g.close()
