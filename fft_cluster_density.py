# cluster from file
# DBSCAN implementation
import os
import glob
import random
import numpy as np
import pickle
from tqdm import tqdm

import fft_utils

# fetch input files
data_dir = '/fs/nexus-scratch/rhaworth/hmp-mini/'
out_dir = '/fs/nexus-scratch/rhaworth/output/'
files = glob.glob(os.path.join(data_dir, '*.txt'))
print(len(files), 'sequences found')

# DBSCAN

# set parameters
minPts = 10
epsilon = 0.7 # min normalized correlation
max_len = 4096 # unused i guess

# TODO: use more memory to make this faster, store distances or whatever

# initialize
labels = np.zeros([len(files)], type=int) # 0 = undefined, -1 = noise, else = cluster label
cluster_count = 0

# define neighborhood query function
def get_neighbors(files, queryseq, epsilon):
    neighbors = []
    for i, file in enumerate(files):
        seq = fft_utils.seq_from_file(file)
        if np.correlate(seq, queryseq) >= epsilon:
            neighbors.append(i)
    return neighbors


# run slow DBSCAN
for i, file in enumerate(tqdm(files)):
    # skip labelled points
    if labels[i] != 0:
        continue

    # get sequence, list of neighbors
    seq = fft_utils.seq_from_file(file)
    neighbors = get_neighbors(files, seq, epsilon)

    # skip non-core points
    if len(neighbors) < minPts:
        labels[i] = -1
        continue

    # we've found a core point
    # iterate current cluster, set label
    cluster_count += 1
    labels[i] = cluster_count

    # find other cluster members
    for point in neighbors:
        # if set to noise, assign cluster label
        if labels[i] == -1:
            labels[i] = cluster_count
        
        # if previously processed, skip
        if labels[i] != 0:
            continue

        # for unprocessed points, set label to cluster_count
        labels[i] = cluster_count

        # get neighbors 
        seq = fft_utils.seq_from_file(file)
        new_neighbors = get_neighbors(files, seq, epsilon)

        # if core point, append new neighbors
        # TODO: make this a set?
        if len(new_neighbors) >= minPts:
            neighbors = neighbors + new_neighbors

# TODO: consider labelling border points as noise

# compute cluster membership for entire dataset
# TODO: idk maybe calculate the support or something

print(np.unique(labels, return_counts=True))

# write labels
with open(os.path.join(out_dir, 'labels.pickle'), 'wb') as f:
    pickle.dump(labels, f)