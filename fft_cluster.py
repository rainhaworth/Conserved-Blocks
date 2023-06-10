# cluster from file
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

# k-medoids w/ mini-batch
# this should probably have some sort of optimizer

# set parameters
batch_size = 5000
iterations = 10
k = 5
max_len = 4096

# initialize
buffer = np.empty([batch_size, max_len])
centroids = np.empty([k, max_len])
random.seed(0)

# populate centroids
for i in range(k):
    centroids[i] = fft_utils.seq_from_file(files[random.randint(0, len(files)-1)])

# run k-means
for i in range(batch_size*iterations):
    # if buffer is full, run k-means
    if i != 0 and i % batch_size == 0:
        print('iteration', i // batch_size)

        cross_corr = np.zeros((batch_size, k))
        for x in range(batch_size):
            for y in range(k):
                cross_corr[x, y] = np.correlate(buffer[x], centroids[y])


        # find max of each column of correlation matrix
        nearest_centroids = np.argmax(cross_corr, axis=1)
        
        # update centroids
        for x in range(0,k):
            # get cluster members, handle special cases
            cluster_idxs = np.where(np.equal(nearest_centroids, x))
            cluster_members = buffer[cluster_idxs]
            if len(cluster_idxs[0]) == 0:
                # if we find literally nothing just try a new one i guess
                # this should probably have some more sophisticated logic if we're doing it
                centroids[x] = fft_utils.seq_from_file(files[random.randint(0, len(files)-1)])
                continue
            if len(cluster_idxs[0]) == 1:
                centroids[x] = cluster_members
                continue

            # get medoid
            # this seems to always converge to 1 cluster lol
            # calculate xcorr on cluster members
            cluster_cross_corr = np.zeros((len(cluster_members), len(cluster_members)))
            for a in range(len(cluster_members)):
                for b in range(len(cluster_members)):
                    cluster_cross_corr[a, b] = np.correlate(cluster_members[a], cluster_members[b])
            medoid_idx = np.argmax(np.sum(cluster_cross_corr, axis=0))
            centroids[x] = cluster_members[medoid_idx]

            # compute support
            cc = np.zeros((batch_size))
            for i in range(len(cluster_members)):
                cc[i] = np.correlate(cluster_members[i], centroids[x])
            print('cluster', x, 'support =', np.mean(cc))

        # don't need to clear buffer bc we will jsut overwrite it

    # populate buffer
    buffer[(i-k) % batch_size] = fft_utils.seq_from_file(files[random.randint(0, len(files)-1)])

# compute cluster membership for entire dataset
membership = []
totals = [0] * k
for file in tqdm(files, 'computing membership'):
    seq = fft_utils.seq_from_file(file)
    cross_corr = np.empty((k))
    for i in range(k):
        cross_corr[i] = np.correlate(seq, centroids[i])
    nearest = np.argmax(cross_corr)
    totals[nearest] += 1
    membership.append((nearest, cross_corr[nearest]))

print("total membership:", totals)

with open(os.path.join(out_dir, 'membership.pickle'), 'wb') as f:
    pickle.dump(membership, f)
