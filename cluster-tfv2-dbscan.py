import os, sys
import tfv2transformer.input as dd
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf

# set max length
max_len = 4096

itokens, otokens = dd.LoadKmerDict('./utils/8mers.txt')
#gen = dd.KmerDataGenerator('./data-tmp/', itokens, otokens, batch_size=1, max_len=max_len)

# make indexed list of all sequences
seqs = dd.DataIndex('/fs/nexus-scratch/rhaworth/synth/', itokens, otokens, max_len=max_len, fasta=True)

from tfv2transformer.transformer_sparse import Transformer, LRSchedulerPerStep

d_model = 512
block_size = 64
s2s = Transformer(itokens, otokens, len_limit=512, d_model=d_model, d_inner_hid=512, \
                   n_head=8, layers=2, length=max_len, block_size=block_size, dropout=0.1)

mfile = '/fs/nexus-scratch/rhaworth/models/tmp.model.h5'

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try: s2s.model.load_weights(mfile)
except: print('No model file found at', mfile)

# make encoder-only model
s2s.make_encode_model()

# copy pasted from fft/fft_cluster_block.py
# DBSCAN
import math, random, pickle
from tqdm import tqdm

# set parameters
out_dir = '/fs/nexus-scratch/rhaworth/clusterout/'
minPts = 5
epsilon = 0.1 # max dist
sample_rate = 0.01

# initialize
#labels = np.zeros([len(files)], dtype=int) # 0 = undefined, -1 = noise, else = cluster label
cluster_count = 0
random.seed(0)

# SNG-DBSCAN
# problems i notice:
    # don't have len(seqs) or len(files) available
    # need to replace distance metric
    # need to compute embeddings for each batch
    # need to set outdir from somewhere
    # how do i iterate over the entire dataset? i don't have infrastructure for that atm
        # need to do that at the end at least
        # how do you iterate over the entire generator?
            # apparently just `for item in gen` does it
    # should i subsample or mini batch?

# construct graph
# initialize with dict comprehension
# honestly i should just write a better mini-batch algorithm
    # sample randomly from different files
        # add a parameter for how many files to split the batch across
        # this is a future step; for simulated data, treat everything as 1 file
    # i guess we need some indexing scheme anyway so make one
        # if we're already writing one, might as well use it to make this one work first
        # need to be able to get() sequence at specific index
        # also, we need the transformer to be able to fetch the embedding for that sequence
    # empty graph, populate in loop
    # replace for i in tqdm with for item in gen
    # see TODO notes

# TODO: implement for split across multiple files
seqcount = seqs.len()
labels = np.zeros([seqcount], dtype=int)
graph = {key: set() for key in range(seqcount)}
for i in tqdm(range(seqcount), 'constructing neighbor graph'):
    # get model output
    pred_i = s2s.encode_model.predict(seqs.get(i), batch_size=1, steps=1, verbose=0)
    points_visited = [i]
    for _ in range(math.ceil(sample_rate * seqcount)):
        # TODO: this is fine but prob needs to be fixed
            # maybe just use random.sample(); i don't think we need to avoid visiting the same point in multiple iterations
            # this is for mini-batch ^
        # sample random indices; ensure unique points and j != i
        j = i
        while j in points_visited:
            j = random.randint(0, seqcount-1)
        points_visited.append(j)

        # get model output for seqs[j]
        pred_j = s2s.encode_model.predict(seqs.get(j), batch_size=1, steps=1, verbose=0)

        # compute distance; pred_x[1] = enc_output
        # just use cosine loss for now and push values from [-1, 1] to [0, 1]
        cos_loss = -tf.reduce_sum([tf.math.l2_normalize(tf.squeeze(pred_i)), tf.math.l2_normalize(tf.squeeze(pred_j))])
        dist = (cos_loss + 1) / 2

        # TODO: check all clusters
        if dist <= epsilon:
            # update both parts of graph
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
for i in tqdm(range(seqcount), 'labelling all points'):
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
# TODO: add more info to filename
with open(os.path.join(out_dir, 'labels-' + str(minPts) + '-' + str(epsilon) + '.pickle'), 'wb') as f:
    pickle.dump(labels, f)