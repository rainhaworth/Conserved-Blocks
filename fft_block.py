# find conserved block for each cluster
import os
import glob
import pickle
import numpy as np
import fft_utils

# fetch input files
data_dir = '/fs/nexus-scratch/rhaworth/hmp-mini/'
out_dir = '/fs/nexus-scratch/rhaworth/output/'
files = glob.glob(os.path.join(data_dir, '*.txt'))
print(len(files), 'sequences found')

membership = []
with open(os.path.join(out_dir, 'membership.pickle'), 'rb') as f:
    membership = pickle.load(f)

# split membership into cluster and support
cluster, support = zip(*membership)

# find list of unique clusters
cluster_ids = np.unique(cluster)

# handle each cluster
for id in cluster_ids:
    # get indices; extract from tuple
    idxs = np.where(np.equal(cluster, id))[0]

    # algorithm:
    # 1. choose medoid as reference sequence, align all other sequences, store offsets
    # 2. get original strings, shift/prune according to offset, store in array
    # 3. iterate over chars and compute best nucleotide + support
    # 4. print, optionally write to fastq

    # find centroid location relative to idxs; raw index = idxs[centroid_idx]
    centroid_idx = np.argmax([support[i] for i in idxs])

    # get subset of files for this cluster
    cluster_files = [files[i] for i in idxs]
    centroid = fft_utils.seq_from_file(cluster_files[centroid_idx])

    # (1) compute offsets
    offsets = np.empty(len(idxs), dtype=int)
    drop = []
    for i, file in enumerate(cluster_files):
        if i == centroid_idx:
            offsets[i] = 0
        else:
            seq = fft_utils.seq_from_file(file)
            xcorr = np.correlate(seq, centroid, mode='same')
            # store offset
            offsets[i] = np.argmax(xcorr)
            # drop samples where xcorr == 0, or alternatively xcorr < some threshold
            if xcorr[offsets[i]] == 0:
                drop.append(i)

    # (2) get original strings, use offset to get substring, store in array
    seq_strs = []
    max_str_len = 0
    for i, file in enumerate(cluster_files):
        if i in drop:
            continue

        with open(file, 'r') as f:
            seq_str = f.read()

        o = offsets[i]
        substr = seq_str[o:]
        if len(substr) != 0:
            seq_strs.append(substr)
            if len(substr) > max_str_len:
                max_str_len = len(substr)
    
    #print(len(seq_strs))

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
        
        max = np.argmax(scores)

        if max == 0:
            block_str += 'A'
        elif max == 1:
            block_str += 'C'
        elif max == 2:
            block_str += 'G' 
        else:
            block_str += 'T'
        
        nucleotide_support.append(scores[max] / len(seq_strs))

    # (4) print
    # TODO: (optionally) prune string when support falls below a threshold
    # TODO: (optionally) write supports in fastq format
    print(block_str)
    print(nucleotide_support)