import numpy as np
import matplotlib.pyplot as plt
import pyemma
import mdtraj as md
import pprint
from six.moves import xrange
import re
import os
import shutil
import argparse
import time
import gzip
from load_data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle




data_traj1 = [('../DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/2F4K-0-c-alpha-'+str(i).zfill(3)+'.dcd') for i in range(7)]

data_traj = data_traj1

CA_dis = pyemma.coordinates.featurizer('2f4k_ca.pdb')
top = md.load_pdb('2f4k_ca.pdb')
top = top.topology
indices = top.select('name CA')

import itertools
pairs = list(itertools.product(indices,indices))
feat = pyemma.coordinates.featurizer(top)
feat.add_distances(pairs, periodic=False)
print(feat.dimension())
data = pyemma.coordinates.load(data_traj, feat)
all_data = np.concatenate(([data[i] for i in range(7)]))

print(all_data.shape)
matrix = all_data.reshape((all_data.shape[0],35,35,1))


np.random.seed(42)
idx = np.arange(matrix.shape[0])
np.random.shuffle(idx)
train_n = int(len(idx)*0.8)
train_data = matrix[idx[0:train_n]]
valid_data = matrix[idx[train_n:]]


scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape([len(train_data),35*35]))
valid_scaled = scaler.transform(valid_data.reshape([len(valid_data), 35*35]))

train_scaled = train_scaled.reshape(len(train_scaled),35,35,1)
valid_scaled = valid_scaled.reshape(len(valid_scaled),35,35,1)



t = md.load_dcd(data_traj[0],top='2f4k_ca.pdb')
for i in range(1,7):
    a = md.load_dcd(data_traj[i],top='2f4k_ca.pdb')
    t = t.join(a)

#atoms_to_keep = [a.index for a in t.topology.atoms if a.name=='CA']
#t.restrict_atoms(atoms_to_keep)

t2 = md.load_pdb('2f4k_ca.pdb')

rmsds = md.rmsd(t,t2,0)
rmsds = rmsds*10

with open('rmsds.npy','wb') as f:
    np.save(f,rmsds)

#quit()

#with open('rmsds.npy','rb') as f:
#    rmsds = np.load(f)

rmsd_train = rmsds[idx[0:train_n]]
rmsd_valid = rmsds[idx[train_n:]]

train_scaled_padded = np.zeros((len(train_scaled),36,36,1))
train_scaled_padded[:,1:,1:] = train_scaled

valid_scaled_padded = np.zeros((len(valid_scaled),36,36,1))
valid_scaled_padded[:,1:,1:] = valid_scaled


x_train = Dataset(train_scaled_padded, rmsd_train)
x_valid = Dataset(valid_scaled_padded, rmsd_valid)

total_scaled = scaler.fit_transform(matrix.reshape(len(matrix), 35*35))
total_scaled = total_scaled.reshape(len(total_scaled),35,35,1)
total_scaled_padded = np.zeros((len(total_scaled),36,36,1))
total_scaled_padded[:,1:,1:] = total_scaled

# pass the labels as indeces to later recover which datapoint belongs to what time
x_total = Dataset(total_scaled_padded, rmsds)

batch_size = 2500
learning_rate = 0.001
epochs = 100
num_layers = 2
hidden_dim = 128
z_dim = 3
temperature = 0.05
K = 6
restore= 0
r_nent = 1
r_label = 1
r_recons = 1

from real_Conv_GMVAE_gumbel import ConvGaussianMixtureVAE
cgmvae = ConvGaussianMixtureVAE(36, 36, r_nent=r_nent, r_label=r_label, r_recons=r_recons,K_clusters=K, restore=restore, dense_n=64, batch_size=batch_size, z_dim=z_dim, filters=[64,64,32], k_sizes=[3,3,3], paddings=['SAME','SAME','SAME'], stride=[1,1,1], pool_sizes=[2,2,1], learning_rate=0.001, epochs=epochs)


train_dic, valid_dic = cgmvae.train(x_train, x_valid)

with open('train_dic.pickle','wb') as handle:
    pickle.dump(train_dic,handle)

with open('valid_dic.pickle','wb') as handle:
    pickle.dump(valid_dic,handle)

f, ax = plt.subplots(4,1, figsize=(10,8))
ax[0].plot(train_dic['loss'])
ax[0].set_ylabel('total loss',fontweight='bold')
ax[1].plot(train_dic['ent'])
ax[1].set_ylabel('cross-entropy',fontweight='bold')
ax[2].plot(train_dic['label'])
ax[2].set_ylabel('labeled loss',fontweight='bold')
ax[3].plot(train_dic['recons'])
ax[3].set_ylabel('reconstruction-loss',fontweight='bold')
ax[3].set_xlabel('epochs',fontweight='bold')
ax[0].set_title('train')
f.savefig('train.png')


f, ax = plt.subplots(4,1, figsize=(10,8))
ax[0].plot(valid_dic['loss'])
ax[0].set_ylabel('total loss',fontweight='bold')
ax[1].plot(valid_dic['ent'])
ax[1].set_ylabel('cross-entropy',fontweight='bold')
ax[2].plot(valid_dic['label'])
ax[2].set_ylabel('labeled loss',fontweight='bold')
ax[3].plot(valid_dic['recons'])
ax[3].set_ylabel('reconstruction-loss',fontweight='bold')
ax[3].set_xlabel('epochs',fontweight='bold')
ax[0].set_title('validation set')
f.savefig('validation.png')

z, y, labels, x_data = cgmvae.generate_embedding(x_total)
labels = np.array(labels)
z = np.array(z)
y = np.array(y)

n = x_total.num_batches(batch_size)
labels = labels.reshape(n*batch_size)
y = y.reshape(n*batch_size,K)
z = z.reshape(n*batch_size,z_dim)


with open('z_tot.npy','wb') as f:
    np.save(f,z)

with open('y_tot.npy','wb') as f:
    np.save(f,y)

with open('labels_tot.npy','wb') as f:
    np.save(f,labels)

quit()

y_m = []
for i in range(len(y)):
    y_m.append(np.argmax(y[i,:]))


y_m = np.array(y_m)
####################################
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE as TSNE
plt.set_cmap('jet')

y_m_t = []
inds = []
thresh = 0.95
for i in range(len(y)):
    if y[i].max()>thresh:
        y_m_t.append(np.argmax(y[i]))
        inds.append(i)

y_m_t = np.array(y_m_t)
z_ind = z[inds]
labels_ind = labels[inds]


sel = np.random.randint(0,len(z_ind),50000)
z_sel = z_ind[sel]
labels_sel = labels_ind[sel]
y_m_t_ind = y_m_t[sel]

tsne = TSNE(n_components=2, n_jobs=-1,verbose=1, init='pca')
#var3d=z
var2d = tsne.fit_transform(z_sel)


with open('var2d_zsel.npy','wb') as f:
    np.save(f,var2d)

with open('labels_sel.npy','wb') as f:
    np.save(f,labels_sel)

with open('y_m_t_ind.npy','wb') as f:
    np.save(f,y_m_t_ind)

plt.figure(figsize=(12,12))
plt.scatter(var2d[:,0],var2d[:,1],c=labels_sel)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('RMSD[A]')
plt.savefig('learned_rmsd.png')

plt.figure(figsize=(12,12))
plt.scatter(var2d[:,0],var2d[:,1],c=y_m_t_ind)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Cluster Labels')
plt.savefig('learned_labels.png')

quit()
####################### Thresholding #####################
y_m_t = []
inds = []
thresh = 0.50
for i in range(len(y)):
    if y[i].max()>thresh:
        y_m_t.append(np.argmax(y[i]))
        inds.append(i)

z_ind = z[inds]


plt.figure(figsize=(12,12))
plt.scatter(z_ind[:,0],z_ind[:,1],c=labels[inds])
plt.savefig('learned_rmsd.png')

plt.figure(figsize=(12,12))
plt.scatter(z_ind[:,0],z_ind[:,1],c=y_m_t)
plt.savefig('learned_labels.png')


#fig = plt.figure(figsize=(14,10))
#ax = fig.add_subplot(111, projection='3d')
#myplot = ax.scatter(z_ind[:,0], z_ind[:,1], z_ind[:,2], c=labels[inds])
#fig.savefig('latent_rmsd_z3.png')

#fig = plt.figure(figsize=(14,10))
#ax = fig.add_subplot(111, projection='3d')
#myplot = ax.scatter(z_ind[:,0], z_ind[:,1], z_ind[:,2], c=y_m_t)
#fig.savefig('latent_labels_z3.png')
