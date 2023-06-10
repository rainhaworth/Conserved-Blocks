# find conserved block for each cluster
import os
import sys
import glob
import pickle
import numpy as np
import fft_utils

import random

from scipy import signal

# fetch input files
data_dir = '/fs/nexus-scratch/rhaworth/hmp-mini/'
out_dir = '/fs/nexus-scratch/rhaworth/output/'
files = glob.glob(os.path.join(data_dir, '*.txt'))
print(len(files), 'sequences found')

# get cluster labels
labels_file = 'labels.pickle'
# if command line arg supplied, use for cluster label file
if len(sys.argv) >= 2:
    labels_file = sys.argv[1]

with open(os.path.join(out_dir, labels_file), 'rb') as f:
    labels = pickle.load(f)

# find list of unique clusters
cluster_ids = np.unique(labels)

# handle each cluster
supports = []
for id in cluster_ids:
    # skip noise
    if id == -1:
        continue

    print('cluster', id)
    # get indices
    idxs = np.where(np.equal(labels, id))[0]
    
    # get subset of files for this cluster
    cluster_files = [files[i] for i in idxs]

    # get best sequence; currently randomly selected
    best_idx = random.choice(range(len(idxs)))

    centroid = fft_utils.seq_from_file(cluster_files[best_idx])

    # (1) compute offsets
    offsets = np.empty(len(idxs), dtype=int)
    for i, file in enumerate(cluster_files):
        if i == best_idx:
            offsets[i] = 0
        else:
            seq = fft_utils.seq_from_file(file)
            xcorr = signal.correlate(seq, centroid, mode='full')
            
            # compute offset value from correlation lags
            lags = signal.correlation_lags(seq.size, centroid.size, mode='full')
            lag = lags[np.argmax(xcorr)]
            
            # store offset
            offsets[i] = lag

    # (2) get original strings, use offset to get substring, store in array
    seq_strs = []
    max_str_len = 0
    for i, file in enumerate(cluster_files):
        with open(file, 'r') as f:
            seq_str = f.read()

        o = offsets[i]
        if o > 0:
            substr = seq_str[o:]
        else:
            substr = ('-'*(-o)) + seq_str[:len(seq_str)+o]
        substr = seq_str[o:]
        if len(substr) != 0:
            seq_strs.append(substr)
            if len(substr) > max_str_len:
                max_str_len = len(substr)

    # (3) iterate over chars in seq_strs, compute best nucleotide + support
    block_str = ""
    nucleotide_support = []
    for i in range(max_str_len):
        scores = np.zeros([4], dtype=int)
        for seq in seq_strs:
            if len(seq) <= i:
                continue

            if seq[i] == 'A':
                scores[0] += 1
            elif seq[i] == 'C':
                scores[1] += 1
            elif seq[i] == 'G':
                scores[2] += 1
            elif seq[i] == 'T':
                scores[3] += 1
        
        m = np.argmax(scores)

        if m == 0:
            block_str += 'A'
        elif m == 1:
            block_str += 'C'
        elif m == 2:
            block_str += 'G' 
        else:
            block_str += 'T'
        
        nucleotide_support.append(scores[m] / len(seq_strs))

    # (4) print
    print(block_str[:100])
    print(np.mean(nucleotide_support), 'quartiles:',
            np.mean(nucleotide_support[:1024]),
            np.mean(nucleotide_support[1024:2048]),
            np.mean(nucleotide_support[2048:3072]),
            np.mean(nucleotide_support[3072:]))

    supports.append(np.mean(nucleotide_support))

    # write to pickle file
    with open(os.path.join(out_dir, 'block_cluster_' + str(id) + '.pickle'), 'wb') as f:
        tup = (block_str, nucleotide_support)
        pickle.dump(tup, f)

print(supports)
