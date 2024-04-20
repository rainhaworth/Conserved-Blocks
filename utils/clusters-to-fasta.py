# output fasta file containing all cluster members
import os
import csv
import model.input as dd
import numpy as np
import time
from math import comb # for computing max number of hits

# set global max length, batch size, and k
max_len = 4096
batch_size = 32
k = 4
d_model = 128

itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = dd.gen_simple_block_data_binary(max_len=max_len, min_len=max_len//4, batch_size=batch_size, tokens=itokens, k=k)
#gen = dd.KmerDataGenerator('/fs/nexus-scratch/rhaworth/hmp-mini/', itokens, otokens, batch_size=4, max_len=max_len)

dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'
cluster_iter = 9 # iteration to look at

dataset = dd.DataIndex(dataset_path, itokens, otokens, k=k, max_len=max_len, fasta=True, metadata=True)

cluster_file = os.path.join(results_dir, 'iter-' + str(cluster_iter) + '-clusters.csv')

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

# get full metadata string for each cluster element
cluster_mds = []
for cl in cluster_idxs:
    mds = []
    for e in cl:
        mds.append(dataset.getmd(e)[:-1])
    cluster_mds.append(mds)

# output fasta file
for i in range(len(cluster_mds)):
    # i = cluster number
    for j in range(len(cluster_mds[i])):
        print('>' + cluster_mds[i][j] + ' cluster=' + str(i))
        print(dataset.index[cluster_idxs[i][j]])
    