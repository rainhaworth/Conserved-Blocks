# inspect clusters
import os
import csv # for writing data
import tfv2transformer.input as dd
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

# make combined list of indices of all sequences in any cluster
all_cluster_idxs = []
for cl in cluster_idxs:
    all_cluster_idxs.extend(cl)

# get full metadata string for each cluster element
cluster_mds = []
for cl in cluster_idxs:
    mds = []
    for e in cl:
        mds.append(dataset.getmd(e)[:-1])
    cluster_mds.append(mds)

all_cluster_mds = []
for cl in cluster_mds:
    all_cluster_mds.extend(cl)

# get list of all pairs we hope to find
pairs = []
for cl in cluster_mds:
    pairs_cl = []
    clen = len(cl)
    for i in range(clen):
        for j in range(i+1, clen):
            pairs_cl.append((cl[i], cl[j]))
    pairs.append(pairs_cl)

# iterate over reduced blast file
hits = [[] for _ in range(len(cluster_idxs))] # track hit locations
maxhits = [len(x) for x in pairs] # compute maximum number of hits for each cluster
with open(blast_red_file, 'r') as f:
    lastquery = ''
    for line in f:
        # get metadata for this line if applicable
        if len(line) < 2:
            continue
        elif 'Query=' in line:
            lastquery = line[7:-1]
            continue
        else:
            md =' '.join(line.split()[:-2]) # remove score numbers and whitespace

        # check whether the sequence on this line AND the last query sequence is in any cluster
        if md in all_cluster_mds and lastquery in all_cluster_mds:
            # figure out which cluster it's in
            clust_num = -1
            for i, cl in enumerate(cluster_mds):
                if md in cl:
                    clust_num = i
                    break
            # this shouldn't happen
            if clust_num == -1:
                print('fatal error: element', md, 'found in all clusters but no specific cluster')
                break
            
            # check whether query is in the same cluster
            # avoid duplicate hits by checking pairs and pruning as we go
            for i, pair in enumerate(pairs[clust_num]):
                if lastquery in pair and md in pair:
                    #print(clust_num, '\t', line[:-1]) # checking score vs cluster num
                    hits[clust_num].append((cluster_mds[clust_num].index(pair[0]), cluster_mds[clust_num].index(pair[1])))
                    pairs[clust_num].pop(i)

# print number of hits
print('summary:')
for i in range(len(hits)):
    print('cluster', i, ':', len(hits[i]), '/', maxhits[i], '({:.2f}%)'.format(100 * len(hits[i]) / maxhits[i]), hits[i])