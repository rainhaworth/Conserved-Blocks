# cluster from file
# DBSCAN implementation
import os
import sys
import glob
import random
import math
import numpy as np
import pickle
from tqdm import tqdm
from scipy import signal

import fft_utils

# fetch input files
data_dir = '/fs/nexus-scratch/rhaworth/hmp-mini/'
out_dir = '/fs/nexus-scratch/rhaworth/output/'
files = glob.glob(os.path.join(data_dir, '*.txt'))
print(len(files), 'sequences found')

# DBSCAN

# set parameters
minPts = 10
epsilon = 0.8 # min normalized correlation
sample_rate = 100 / len(files)
max_len = 4096

# set method string
method = 'ML-DSP'

# if given, grab params from command line
# order: minPts epsilon method
if len(sys.argv) >= 4:
    minPts = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    method = sys.argv[3]


# initialize
labels = np.zeros([len(files)], dtype=int) # 0 = undefined, -1 = noise, else = cluster label
cluster_count = 0
random.seed(0)

# compute max correlation for re-normalization, which unfortunately we need to do
# band-aid solution: after first run, find index where highest correlation occurs, insert it here
_seq = fft_utils.seq_from_file(files[14627], method)
max_corr = signal.correlate(_seq, _seq, 'valid')[0]
print(max_corr)

# load all sequences from files + pre-compute fft
seqs = np.zeros([len(files), max_len], dtype=float)
for i, file in enumerate(tqdm(files, 'pre-computing fft')):
    seqs[i] = fft_utils.seq_from_file(file, method)

# SNG-DBSCAN

# construct graph
# initialize with dict comprehension
graph = {key: set() for key in range(len(seqs))}
for i in tqdm(range(len(seqs)), 'constructing neighbor graph'):
    points_visited = [i]
    for _ in range(math.ceil(sample_rate * len(seqs))):
        # sample random indices; ensure unique points and j != i
        j = i
        while j in points_visited:
            j = random.randint(0, len(seqs)-1)
        points_visited.append(j)

        # compare; re-normalize
        xcorr = signal.correlate(seqs[i], seqs[j], 'valid')
        if xcorr > max_corr:
            print(xcorr, i, j)
            max_corr = xcorr
        if xcorr / max_corr >= epsilon:
            # update both parts of graphgraph
            graph[i].add(j)
            graph[j].add(i)

# get set of connected components induced by at least minPts vertices
clusters = []
for key in tqdm(graph, 'finding dense clusters'):
    # skip already processed points
    processed = False
    for c in clusters:
        if key in c:
            processed = True
            break
    if processed:
        continue

    # search for clusters of connected core points
    connected = list(graph[key])
    if len(connected) >= minPts:
        for elem in connected:
            new_connected = graph[elem]
            if len(new_connected) >= minPts:
                connected += list(new_connected - set(connected))
        clusters.append(connected)

print(len(clusters), "clusters found")

# iterate over all points and assign labels
# -1 = noise, 0+ = cluster label
for i in tqdm(range(len(seqs)), 'labelling all points'):
    labels[i] = -1

    # check whether point is in cluster
    for j, cluster in enumerate(clusters):
        if i in cluster:
            labels[i] = j
            break
    
    if labels[i] != -1:
        continue

    # if not found, check whether point is connected to cluster
    for j, cluster in enumerate(clusters):
        for elem in graph[i]:
            if elem in cluster:
                labels[i] = j
                break
        if labels[i] != -1:
            break

    # else leave as noise

# could try to compute support but it's weird without a centroid

print(np.unique(labels, return_counts=True))

# write labels
with open(os.path.join(out_dir, 'labels-' + str(minPts) + '-' + str(epsilon) + '-' + method + '.pickle'), 'wb') as f:
    pickle.dump(labels, f)
