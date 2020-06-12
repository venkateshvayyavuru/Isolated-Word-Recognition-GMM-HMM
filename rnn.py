import numpy as np
import matplotlib.pyplot as plt
data=open("text.txt",'r').read()
chars=list(set(data))
data_len,vocab_len=len(data),len(chars)
char_to_index={char:index for index,char in enumerate(chars)}
indx_to_chr={index:char for index,char in enumerate(chars)}
hidden_dim=200
seq_length=40
lr_rate=0.01
g=open("generated_text_40_200dim.txt",'w')
#initialise the model paramteres
Wxh=np.random.randn(hidden_dim,vocab_len)*0.01
Whh=np.random.randn(hidden_dim,hidden_dim)*0.01
bh=np.random.randn(hidden_dim,1)*0.01
Why=np.random.randn(vocab_len,hidden_dim)*0.01
by=np.random.randn(vocab_len,1)

def model(inputs,targets,hprev):
 #Forward Pass
 x,h,z,y={},{},{},{}
 h[-1]=np.copy(hprev)
 loss=0
 for t in range(0,len(inputs)):
  x[t]=np.zeros((vocab_len,1))
  x[t][inputs[t]]=1
  h[t]=np.tanh(np.dot(Wxh,x[t])+np.dot(Whh,h[t-1])+bh)
  z[t]=np.dot(Why,h[t])+by
  y[t]=np.exp(z[t])/np.sum(np.exp(z[t]))
  loss=loss-np.log(y[t][targets[t],0])
 #Backward-pass
 dWxh=np.zeros_like(Wxh)
 dWhh=np.zeros_like(Whh)
 dWhy=np.zeros_like(Why)
 dbh=np.zeros_like(bh)
 dby=np.zeros_like(by)
 dh_next=np.zeros_like(h[0])
 for t in reversed(range(0,len(inputs))):
  dz=np.copy(y[t])
  dz[targets[t]]=dz[targets[t]]-1 ###         y[t]-p[t] softmax plus cross-entropy
  dWhy=dWhy+np.dot(dz,h[t].T)
  dby=dby+dz
  dh=np.dot(Why.T,dz)+dh_next
  da=(1-h[t]*h[t])*dh
  dbh=dbh+da
  dWhh=np.dot(da,h[t-1].T)
  dWxh=np.dot(da,x[t].T)
  dh_next=np.dot(Whh.T,da)
 return loss,dWxh,dWhh,dWhy,dbh,dby,h[len(inputs)-1]
def sampling(input_chunk,h_prev,epoch):
 x,y,h,z={},{},{},{}
 h[-1]=np.copy(h_prev)
 g.write(str(epoch)+"\t")
 for t in range(0,len(input_chunk)):
  x[t]=np.zeros((vocab_len,1))
  x[t][input_chunk[t]]=1
  h[t]=np.tanh(np.dot(Wxh,x[t])+np.dot(Whh,h[t-1])+bh)
  z[t]=np.dot(Why,h[t])+by
  y[t]=np.exp(z[t])/np.sum(np.exp(z[t]))
  print(str(indx_to_chr[np.argmax(y[t])]+" "))
  g.write(str(indx_to_chr[np.argmax(y[t])]+" ")+"\n")
n, p = 0, 0
num_epohs=25
loss_epochs=[]
for epoch in range(0,num_epohs):
 for p in range(0,len(data)-seq_length,seq_length):
  if p+seq_length+1 >len(data) or p==0 :
    h_prev = np.zeros((hidden_dim,1))
  inputs=[char_to_index[ch] for ch in data[p:p+seq_length]]
  targets=[char_to_index[ch] for ch in data[p+1:p+seq_length+1]]
  loss,dWxh,dWhh,dWhy,dbh,dby,h=model(inputs,targets,h_prev)
  for parameter,dparameter in zip([Wxh, Whh, Why, bh, by],[dWxh, dWhh, dWhy, dbh, dby]):
   parameter += -lr_rate * dparameter
 loss=loss/20
 loss_epochs.append(loss)
 g.write("epoch:"+str(epoch)+"\t"+"loss:"+str(loss)+"\n")
 if epoch %10 ==0:
  h=np.zeros((hidden_dim,1))
  inputs=[char_to_index[ch] for ch in data[0:40]] 
  sampling(inputs,h,epoch)
plt.plot(loss_list)
plt.xlabel("EPochs")
plt.ylabel("Loss")
plt.title("EPochs vs LOSS")
plt.savefig("40seq_200_dim_loss_vs_epochs.png")

   
  
