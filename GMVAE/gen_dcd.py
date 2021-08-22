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

with open('labels_tot.npy','rb') as f:
	labels_tot = np.load(f)

with open('z_tot.npy','rb') as f:
	z_tot = np.load(f)

with open('y_tot.npy','rb') as f:
	y_tot = np.load(f)


labels_tot = np.arange(len(z_tot))
y_m_t = []
inds = []
thresh = 0.9
for i in range(len(y_tot)):
    if y_tot[i].max()>thresh:
        y_m_t.append(np.argmax(y_tot[i]))
        inds.append(i)


z_ind = z_tot[inds]
y_m_t = np.array(y_m_t)
labels_ind = labels_tot[inds]

ave = np.zeros((len(np.unique(y_m_t)),3))
for i,n in enumerate(np.unique(y_m_t)):
    ave[i,:] = z_ind[y_m_t==n].mean(axis=0)


snapshots = []
unique = sorted(np.unique(y_m_t))
for i in range(len(np.unique(y_m_t))):
    snapshots.append(np.argsort(((z_ind-ave[i,:])**2).sum(axis=1))[:5000])

snaps = np.array(snapshots)

ml_samples = labels_ind[snaps]
#n = 1000 # njumber of samples

##samples = []
#unique = sorted(np.unique(y_m_t))
#for i,n in enumerate(unique):
#    sample = labels_ind[y_m_t==n]
#    inds_sample = np.random.randint(0,len(sample),1000)
#    samples.append(sample[inds_sample])

#ml_samples = np.array(samples)

all_samples = []
for row in range(len(ml_samples)):
    a,b  = (ml_samples[row]//10000,ml_samples[row]%10000)
    ab = np.array([[a[i],b[i]] for i in range(5000)])
    all_samples.append(ab)

all_samples = np.array(all_samples)
print(all_samples.shape)
data_traj = [('../DESRES-Trajectory_2F4K-0-protein/2F4K-0-protein/2F4K-0-protein-'+str(i).zfill(3)+'.dcd') for i in range(63)]
trj_source = pyemma.coordinates.source(data_traj, top='2f4k.pdb')

pyemma.coordinates.save_trajs(trj_source, all_samples, outfiles=['gmm_{}_samples.dcd'.format(n) for n in range(len(unique))])
#../DESRES-Trajectory_2F4K-0-protein/2F4K-0-protein/
top = md.load_pdb('2f4k.pdb')
for i in range(6):
    t = 'gmm_'+str(i)+'_samples.dcd'
    trj = md.load_dcd(t,top=top)
    ca = [a.index for a in trj.topology.atoms if a.name=='CA']
    avg = trj.xyz.mean(axis=0)
    top.xyz = avg
    rmsd = md.rmsd(trj, top, atom_indices=ca)
    indices = rmsd.argsort()[:50]
    name='cluster_'+str(i)+'.dcd'
    namepdb='cluster_'+str(i)+'.pdb'
    trj[indices].save_dcd(name)
    top.xyz = trj[indices[1]].xyz
    trj[indices[1]].save_pdb(namepdb)




