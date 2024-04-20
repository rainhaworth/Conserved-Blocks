# inspect clusters
import os
import csv # for writing data
import model.input as dd
import numpy as np
import time
from math import comb # for computing max number of hits
import argparse

parser = argparse.ArgumentParser()
# model
parser.add_argument('--max_len', default=4096, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
# eval
parser.add_argument('-c', '--cluster_batch_size', default=512, type=int)
# output
parser.add_argument('-o', '--output', type=str, choices={'summary', 'allhits', 'gephicsv'}, default='summary')

args = parser.parse_args()

# set global max length, batch size, and k
max_len = args.max_len
batch_size = args.batch_size
k = args.k
d_model = args.d_model

itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = dd.gen_simple_block_data_binary(max_len=max_len, min_len=max_len//4, batch_size=batch_size, tokens=itokens, k=k)
#gen = dd.KmerDataGenerator('/fs/nexus-scratch/rhaworth/hmp-mini/', itokens, otokens, batch_size=4, max_len=max_len)

dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'
blast_red_file = 'prem-micro-blast-reduced.txt'

dataset = dd.DataIndex(dataset_path, itokens, otokens, k=k, max_len=max_len, fasta=True, metadata=True)

cluster_file = os.path.join(results_dir, 'batchsz-' + str(args.cluster_batch_size) + '-clusters.csv')
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
        md = dataset.getmd(e)
        if md is None:
            print('no metadata found for sequence', e)
        else:
            mds.append(md[:-1])
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
    if args.output == 'summary':
        # don't print full list of hits
        print('cluster', i, ':', len(hits[i]), '/', maxhits[i], '({:.2f}%)'.format(100 * len(hits[i]) / maxhits[i]))
    elif args.output == 'allhits':
        # print full list of hits
        print('cluster', i, ':', len(hits[i]), '/', maxhits[i], '({:.2f}%)'.format(100 * len(hits[i]) / maxhits[i]), hits[i])
    elif args.output == 'gephicsv':
        # make gephi edge list CSV
        for edge in hits[i]:
            print(str(cluster_idxs[i][edge[0]]) + ';' + str(cluster_idxs[i][edge[1]]))

# across all hits
if args.output == 'summary' or args.output == 'allhits':
    hit_nums = [len(hits[i]) for i in range(len(hits))]
    print('total:', np.sum(hit_nums), '/', np.sum(maxhits), '({:.2f}%)'.format(100 * np.sum(hit_nums) / np.sum(maxhits)))