import os
import sys
import glob
import random
import math
import numpy as np
import pickle
from tqdm import tqdm

blastfile = '/fs/nexus-scratch/rhaworth/SRS042628-blast-out.txt'

# fetch input files
data_dir = '/fs/nexus-scratch/rhaworth/hmp-mini/'
out_dir = '/fs/nexus-scratch/rhaworth/output/'
files = glob.glob(os.path.join(data_dir, '*.txt'))
print(len(files), 'sequences found')

# get labels; use first command line arg if supplied
labels_file = 'labels.pickle'
if len(sys.argv) >= 2:
    labels_file = sys.argv[1]
with open(os.path.join(out_dir, labels_file), 'rb') as f:
    labels = pickle.load(f)

# get unique sequences in clusters
clusters = np.unique(labels)

cluster_seqs = []
for c in clusters:
    if c == -1:
        continue

    print("cluster", c)

    # get files in cluster
    idxs = np.where(np.equal(labels, c))[0]
    cluster_files = [files[i] for i in idxs]

    # translate filenames into sequence names
    unique_files = set()
    for file in cluster_files:
        s = os.path.basename(file)
        #print(s)
        s_idx = s.find('.bz2')
        unique_files.add(s[s_idx+5:-4])
    print(len(unique_files))
    cluster_seqs.append(unique_files)

# compute support from blast

support = [0] * (len(clusters) - 1)
with open(blastfile, 'r') as f:
    curr_seq = ''
    curr_cluster = -1
    for line in f:
        if 'Query=' in line:
            seq = line[7:-1]
            for i, c in enumerate(cluster_seqs):
                if seq in c:
                    curr_seq = seq
                    curr_cluster = i
                    break
        elif curr_seq != '':
            if len(line.split()) < 2:
                continue
            if 'Length= ' in line:
                continue
            if 'Sequences' in line:
                continue
            if 'Score' in line:
                continue
            if line[0] == '>':
                curr_seq = ''
                continue
            seq = line.split()
            seq = seq[0]

            if seq != curr_seq and seq in cluster_seqs[curr_cluster]:
                support[curr_cluster] += 1

print('support:', support)
