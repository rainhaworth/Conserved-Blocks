# cluster from file
# DBSCAN implementation
import os
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
minPts = 20
epsilon = 0.9 # min normalized correlation
sample_rate = math.log2(len(files)) / len(files)
max_len = 4096

# TODO: look into more scalable implementations

# initialize
labels = np.zeros([len(files)], dtype=int) # 0 = undefined, -1 = noise, else = cluster label
cluster_count = 0
random.seed(0)

# ok what if we just load all the sequences
seqs = np.zeros([len(files), max_len], dtype=float)
for i, file in enumerate(tqdm(files, 'pre-computing fft')):
    seqs[i] = fft_utils.seq_from_file(file)


# define neighborhood query function
def get_neighbors(seqs, query, epsilon):
    neighbors = []
    for i in range(len(seqs)):
        if i == query:
            continue
        if signal.correlate(seqs[i], seqs[query], 'valid') >= epsilon:
            neighbors.append(i)
    return neighbors


# run slow DBSCAN
"""
for i in tqdm(range(len(seqs)), 'clustering'):
    # skip labelled points
    if labels[i] != 0:
        continue

    # get sequence, list of neighbors
    #seq = fft_utils.seq_from_file(file)
    neighbors = get_neighbors(seqs, i, epsilon)

    # skip non-core points
    if len(neighbors) < minPts:
        labels[i] = -1
        continue

    # we've found a core point
    # iterate current cluster, set label
    cluster_count += 1
    labels[i] = cluster_count

    # okay fun idea
    # instead of finding the other cluster members, we assume we have all the ones that matter
    # and we just update them and move on
    # if this works, go back over and see if any clusters can be merged at the end
    for point in neighbors:
        labels[point] = cluster_count

    # find other cluster members
    members = set(neighbors)
    print(len(members))
    while len(members) > 0:
        point = members.pop()
        # if set to noise, assign cluster label
        if labels[point] == -1:
            labels[point] = cluster_count
        
        # if previously processed, skip
        if labels[point] != 0:
            continue

        # for unprocessed points, set label to cluster_count
        labels[point] = cluster_count

        # get neighbors 
        #seq = fft_utils.seq_from_file(file)
        neighbors = get_neighbors(seqs, point, epsilon)

        # if core point, append any new neighbors to members
        if len(neighbors) >= minPts:
            members |= set(neighbors)
            print(len(members))
"""

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

        # compare
        if signal.correlate(seqs[i], seqs[j], 'valid') >= epsilon:
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

    # search for core points; populate clusters
    connected = list(graph[key])
    if len(connected) >= minPts:
        for elem in connected:
            connected += list(graph[elem] - set(connected))
        clusters.append(connected)

print(len(clusters), clusters)

# iterate over all points and assign labels
# -1 = noise, 0+ = cluster label
# TODO: figure out what to do with border points; for now they are assigned to the smallest-index cluster they appear in
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


# TODO: consider labelling border points as noise

# compute cluster membership for entire dataset
# TODO: calculate support somehow? we don't have a centroid tho

print(np.unique(labels, return_counts=True))

# write labels
with open(os.path.join(out_dir, 'labels.pickle'), 'wb') as f:
    pickle.dump(labels, f)