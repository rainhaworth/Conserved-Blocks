# use binary classifier (e.g. SkewedAttention) for clustering
# inspired by SCRAPT
import os, sys
import csv # for writing data
import tfv2transformer.input as dd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm.auto import tqdm
import time
import argparse

# command line args
parser = argparse.ArgumentParser()
# model
parser.add_argument('--max_len', default=4096, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
# clustering
parser.add_argument('--cluster_batch_size', default=512, type=int)
parser.add_argument('--iterations', default=15, type=int)
args = parser.parse_args()

# set global max length, batch size, and k
max_len = args.max_len
batch_size = args.batch_size
k = args.k
d_model = args.d_model

# load kmer dict
itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)

# initialize model
from tfv2transformer.skew_attn import SimpleSkewBinary
ssb = SimpleSkewBinary(itokens, d_model=d_model, length=max_len)

# load weights
mfile = '/fs/nexus-scratch/rhaworth/models/skew.model.h5'
ssb.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9)) # can we just load this with no optimizer?
try: ssb.model.load_weights(mfile)
except: print('\nno model found')


### CLUSTERING ###

cluster_batch_size = args.cluster_batch_size
iterations = args.iterations
boundary = 0.9

dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'

dataset = dd.DataIndex(dataset_path, itokens, otokens, k=k, max_len=max_len, fasta=True, metadata=True)
print('dataset size:', dataset.len())
print('data shape:', np.shape(dataset.get(0)))

# set of unclustered indices
unclustered = set(range(dataset.len()))

clusters = [] # list of sets of ints
representatives = [] # list of ints
representatives_prev = set() # avoid searching the entire dataset again for representatives we've seen before

# set start time
start_time = time.time()

# per-iteration metrics
iter_times = []
iter_clust_counts = []
iter_seq_counts = []

for iter in range(iterations):
    print('iteration', iter)

    # 1. get mini-batch

    # get real batch size
    bsz = min(len(unclustered), cluster_batch_size)
    # get random selection of indices
    # TODO: split across files if multiple are present
    batch_idx = np.random.choice(list(unclustered), bsz, replace=False)
    # populate batch
    batch = np.zeros((bsz, max_len))
    for i, idx in enumerate(batch_idx):
        batch[i] = dataset.get(idx)

    # 2. cluster mini-batch

    # get list of all (cluster_batch_size choose 2) unique combinations of samples
    # could use a generator to save on memory, might be necessary for large cluster batches
    unique_pairs = []
    for i in range(bsz):
        for j in range(i+1,bsz):
            unique_pairs.append([i, j])

    # run all predictions
    for i in tqdm(range(0, len(unique_pairs), batch_size), desc='clustering mini-batch'):
        # construct cluster batch
        current_pairs = np.array(unique_pairs[i:min(i+batch_size, len(unique_pairs))])
        current_pairs = current_pairs.T

        seqs_a = batch[current_pairs[0]]
        seqs_b = batch[current_pairs[1]]

        # predict
        preds = ssb.model.predict([seqs_a, seqs_b], batch_size, verbose=0)
        hits = np.where(preds > boundary)

        # update clusters
        for hit in hits[0]:
            x = current_pairs[0][hit]
            y = current_pairs[1][hit]
            # fetch original indices in DataIndex
            x = batch_idx[x]
            y = batch_idx[y]
            if len(clusters) == 0:
                clusters.append(set([x, y]))
            else:
                cluster_found = 0
                for j, cluster in enumerate(clusters):
                    # TODO: use better cluster update rule
                    if x in cluster or y in cluster:
                        clusters[j].add(x)
                        clusters[j].add(y)
                        cluster_found = 1
                # make new cluster if we don't find one
                if cluster_found == 0:
                    clusters.append(set([x, y]))

    print('cluster count:', len(clusters))
    iter_clust_counts.append(len(clusters))

    # 3. get representative sequence for each cluster

    # TODO: try using logits to find best representative
    # for now just pick a random one and reset every time
    representatives = []
    seqs_clustered = 0
    for cluster in clusters:
        # compute seqs_clustered, assuming no duplicates
        seqs_clustered += len(cluster)
        # get element with max multiplicity
        max_multi = 0.0
        max_elem = -1
        for elem_idx in cluster:
            md = dataset.getmd(elem_idx)
            if md is None:
                break
            md = md.split()
            for substr in md:
                if 'multi=' in substr:
                    multi = float(substr[6:])
                    if multi > max_multi:
                        max_multi = multi
                        max_elem = elem_idx
                    break # stop iterating over substrings
                    
        # set random representative if no multiplicity data
        if max_elem == -1:
            representatives.append(np.random.choice(list(cluster)))
        else:
            representatives.append(max_elem)
        # also update unclustered
        unclustered = unclustered.difference(cluster)

    reps_to_check = set(representatives)
    reps_to_check = list(reps_to_check.difference(representatives_prev))
    unclist = list(unclustered)
    
    # skip to next iteration if we have no representatives
    if len(reps_to_check) == 0:
        print('no new cluster representatives found; skipping to next iteration')
        iter_times.append(time.time() - start_time)
        iter_seq_counts.append(seqs_clustered)
        continue

    # 4. iterate over all unclustered data and predict vs representative sequences

    for i in tqdm(range(0, len(unclustered), batch_size), 'searching full dataset'):
        # get batch of unclustered sequences
        x = np.zeros((batch_size, max_len))
        for j, idx in enumerate(unclist[i:i+batch_size]):
            x[j] = dataset.get(idx)

        for rep in reps_to_check:
            # get batch of representatives
            # TODO: generate these outside the inner loop
            y = np.zeros((batch_size, max_len))
            rep_data = dataset.get(rep)
            for j in range(batch_size):
                y[j] = rep_data

            # predict
            preds = ssb.model.predict([x, y], batch_size, verbose=0)
            hits = np.where(preds > boundary)

            # update cluster if we find any hits
            for hit in hits[0]:
                # get cluster number; assume no duplicate representatives
                repnum = representatives.index(rep)

                idx = unclist[i + hit]
                clusters[repnum].add(idx)
                seqs_clustered += 1

    # 5. update list of remaining points in dataset + previously seen representatives

    for cluster in clusters:
        unclustered = unclustered.difference(cluster)
    representatives_prev = representatives_prev.union(set(representatives))

    # 6. record data for this iteration
        
    outfilename = os.path.join(results_dir, 'iter-' + str(iter) + '-clusters.csv')

    with open(outfilename, 'w') as f:
        writer = csv.writer(f)
        # header
        writer.writerow(['ID', 'Rep', 'Members'])
        for i, cluster in enumerate(clusters):
            row = [i, representatives[i]]
            row += list(cluster)
            writer.writerow(row)

    # continue to next iteration
    print('wrote cluster data to', outfilename)
    print('remaining unclustered data:', len(unclustered))
    print('time elapsed:', time.time() - start_time)
    iter_times.append(time.time() - start_time)
    iter_seq_counts.append(seqs_clustered)

# write final clusters to file
outfile1 = os.path.join(results_dir, 'batchsz-' + str(cluster_batch_size) + '-clusters.csv')
with open(outfile1, 'w') as f:
    writer = csv.writer(f)
    # header
    writer.writerow(['ID', 'Rep', 'Members'])
    for i, cluster in enumerate(clusters):
        row = [i, representatives[i]]
        row += list(cluster)
        writer.writerow(row)

# write per-iter metrics to file
outfile2 = os.path.join(results_dir, 'batchsz-' + str(cluster_batch_size) + '-iters.csv')
with open(outfile2, 'w') as f:
    writer = csv.writer(f)
    # header
    writer.writerow(['Iter', 'Time', 'Clusters', 'Seqs'])
    for i in range(len(iter_times)):
        row = [i, iter_times[i], iter_clust_counts[i], iter_seq_counts[i]]
        writer.writerow(row)

print('completed execution. wrote final outputs to files.')