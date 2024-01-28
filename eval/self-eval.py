# inspect clusters with SCA
import os, sys
import csv # for writing data
import tfv2transformer.input as dd
import numpy as np
import time
from math import comb # for computing max number of hits

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

# set global max length, batch size, and k
max_len = 4096
batch_size = 32
k = 4
d_model = 128

# set decision boundary
boundary = 0.9

itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = dd.gen_simple_block_data_binary(max_len=max_len, min_len=max_len//4, batch_size=batch_size, tokens=itokens, k=k)
#gen = dd.KmerDataGenerator('/fs/nexus-scratch/rhaworth/hmp-mini/', itokens, otokens, batch_size=4, max_len=max_len)

# load model

from tfv2transformer.skew_attn import SimpleSkewBinary

ssb = SimpleSkewBinary(itokens, d_model=d_model, length=max_len)

mfile = '/fs/nexus-scratch/rhaworth/models/skew.model.h5'

# load weights
ssb.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9)) # can we just load this with no optimizer?
try: ssb.model.load_weights(mfile)
except: print('\nno model found')

# cluster eval

dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'
blast_red_file = 'prem-micro-blast-reduced.txt'
cluster_iter = 9 # iteration to look at

dataset = dd.DataIndex(dataset_path, itokens, otokens, k=k, max_len=max_len, fasta=True, metadata=True)

cluster_file = os.path.join(results_dir, 'iter-' + str(cluster_iter) + '-clusters.csv')
blast_red_file = os.path.join(results_dir, blast_red_file)

# get list of sequence indices for each cluster
cluster_idxs = []
with open(cluster_file, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            # skip header
            continue
        num = row[0]
        representative = row[1]
        members = row[2:]
        members = [int(x) for x in members]
        cluster_idxs.append(members)

# run predict on all pairs for each cluster
hits = [[] for _ in range(len(cluster_idxs))] # track hit locations
maxhits = [comb(len(x), 2) for x in cluster_idxs] # compute maximum number of hits for each cluster
for i, cl_idx in enumerate(cluster_idxs):
    # make list of pairs
    cl_pairs = []
    for j in range(len(cl_idx)):
        for k in range(j+1,len(cl_idx)):
            cl_pairs.append([j, k])

    # get data for this cluster
    batch = np.zeros((len(cl_idx), max_len))
    for j, idx in enumerate(cl_idx):
        batch[j] = dataset.get(idx)

    # iterate over all pairs for this cluster
    for j in range(0, len(cl_pairs), batch_size):
        current_pairs = np.array(cl_pairs[j:min(j+batch_size, len(cl_pairs))])
        current_pairs = current_pairs.T

        seqs_a = batch[current_pairs[0]]
        seqs_b = batch[current_pairs[1]]

        # predict
        preds = ssb.model.predict([seqs_a, seqs_b], batch_size, verbose=0)
        _hits = np.where(preds > boundary)[0]
        for hit in _hits:
            hits[i].append((current_pairs[0][hit], current_pairs[1][hit]))
            

# print number of hits
print('summary:')
for i in range(len(hits)):
    print('cluster', i, ':', len(hits[i]), '/', maxhits[i], '({:.2f}%)'.format(100 * len(hits[i]) / maxhits[i]), hits[i])