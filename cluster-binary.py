# use binary classifier (e.g. SkewedAttention) for clustering
# inspired by SCRAPT
import os, sys
import csv # for writing data
import tfv2transformer.input as dd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tqdm.auto import tqdm

# set global max length, batch size, and k
max_len = 4096
batch_size = 32 # this is the maximum we can achieve with the current settings
k = 4
d_model = 128

# TODO: add modified dataindex instead of gen
itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = dd.gen_simple_block_data_binary(max_len=max_len, min_len=max_len//4, batch_size=batch_size, tokens=itokens, k=k)
#gen = dd.KmerDataGenerator('/fs/nexus-scratch/rhaworth/hmp-mini/', itokens, otokens, batch_size=4, max_len=max_len)

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num()) # we don't use this here, go back and fix later

#from tfv2transformer.transformer_sparse import LRSchedulerPerStep
from tfv2transformer.skew_attn import SimpleSkewBinary

ssb = SimpleSkewBinary(itokens, d_model=d_model, length=max_len)

mfile = '/fs/nexus-scratch/rhaworth/models/skew.model.h5'

#lr_scheduler = LRSchedulerPerStep(d_model, 4000)
#model_saver = ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

# load weights
ssb.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9)) # can we just load this with no optimizer?
try: ssb.model.load_weights(mfile)
except: print('\nno model found')

#ssb.model.summary()
#if not os.path.isdir('models'): os.mkdir('models')

### CLUSTERING ###
cluster_batch_size = 512
iterations = 10
boundary = 0.9

dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'

# TODO: put real path into this, also make it use only one set of tokens ideally
dataset = dd.DataIndex(dataset_path, itokens, otokens, k=k, max_len=max_len, fasta=True)
print('dataset size:', dataset.len())
print('data shape:', np.shape(dataset.get(0)))

# set of unclustered indices
unclustered = set(range(dataset.len()))

clusters = [] # list of sets of ints
representatives = [] # list of ints
representatives_prev = set() # avoid searching the entire dataset again for representatives we've seen before

# define cluster update function
# TODO: do this, for now just duplicate the code it's fine
#def update_clusters(clusters, hits, x, y):

for iter in range(iterations):
    print('iteration', iter+1)

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

    #batch = tf.random.uniform((cluster_batch_size, max_len), dtype=int)

    # 2. cluster mini-batch

    # get list of all (cluster_batch_size choose 2) unique combinations of samples
    # TODO: could use a generator to save on memory, might be necessary for large cluster batches
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
    
    # skip everything else if no clusters
    if len(clusters) == 0:
        print('no clusters found; skipping to next iteration')
        continue

    # 3. get representative sequence for each cluster

    # TODO: use logits to find best representative
    # for now just pick a random one and reset every time
    representatives = []
    for cluster in clusters:
        representatives.append(np.random.choice(list(cluster)))
        # also update unclustered
        unclustered = unclustered.difference(cluster)

    # 4. iterate over all unclustered data and predict vs representative sequences

    reps_to_check = set(representatives)
    reps_to_check = list(reps_to_check.difference(representatives_prev))
    unclist = list(unclustered)
    for i in tqdm(range(0, len(unclustered), batch_size), 'searching full dataset'):
        # get batch of unclustered sequences
        x = np.zeros((batch_size, max_len))
        for j, idx in enumerate(unclist[i:i+batch_size]):
            x[j] = dataset.get(idx)

        for repnum, rep in enumerate(reps_to_check):
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
                idx = unclist[i + hit]
                clusters[repnum].add(idx)

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