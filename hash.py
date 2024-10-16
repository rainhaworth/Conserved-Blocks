# hash all sequences from .fasta files in directory
import os
import pickle
import model.input as dd
from tqdm import tqdm
import time
import argparse
import tensorflow as tf

from model.input import pad_to_max, HashIndex

# command line args
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lenseq', default=2000, type=int)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-d', '--d_model', default=64, type=int)
parser.add_argument('-z', '--hashsz', default=64, type=int)
parser.add_argument('-n', '--n_hash', default=8, type=int)
parser.add_argument('-b', '--batchsz', default=1024, type=int)
parser.add_argument('-m', '--mfile', default='/fs/nexus-scratch/rhaworth/models/chunkhashdep.model.h5', type=str)
parser.add_argument('-r', '--resultdir', default='/fs/nexus-scratch/rhaworth/output/', type=str)
parser.add_argument('-i', '--indir', default='/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/', type=str)
parser.add_argument('-o', '--outfile', default='hashindex.pickle', type=str)
parser.add_argument('--ext', default='fa', type=str)
args = parser.parse_args()

# set global vars
chunk_size = args.lenseq
k = args.k
d_model = args.d_model
hash_size = args.hashsz
n_hash = args.n_hash

# load kmer dict
itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)

# initialize model
from model.chunk_hash import ChunkMultiHash
ch = ChunkMultiHash(itokens, chunk_size, d_model, hash_size, n_hash)

# load weights
mfile = args.mfile
ch.compile()
try: ch.model.load_weights(mfile)
except: print('\nno model found')
ch.make_inference_model()

### HASHING ###

# TODO: these should be parameters too
dataset_path = args.indir
results_dir = args.resultdir
ext = args.ext

hash_index = HashIndex(dataset_path, n_hash, ext)
print('files found:', len(hash_index.filenames))

# set start time
start_time = time.time()

# set batch size to largest multiple of 2 that fits on GPU
bsz = args.batchsz

# print statement profiling because i still don't know how python profilers work
batch_time = 0.0
predict_time = 0.0
convert_time = 0.0
table_time = 0.0

for fileidx, filename in enumerate(hash_index.filenames):
    print('file', fileidx, ':', filename)

    # read file, get sequences, split into chunks
    chunks = hash_index.chunks_from_file(fileidx, chunk_size, 0.5, k)

    # iterate over chunks
    for idx in tqdm(range(0, len(chunks), bsz)):
        # get current batch of chunk strings, convert to kmers, convert to tensor
        _b = time.time()
        batch = chunks[idx:idx+bsz]
        batch_kmers = [[seq[i:i+k] for i in range(chunk_size - k + 1)] for seq in batch]
        batch_tokens = pad_to_max(batch_kmers, itokens, chunk_size)
        
        # call model to get hash
        # shape: (bsz, n, h)
        _p = time.time()
        hashes_binary = ch.hasher.predict(batch_tokens, verbose=0) # type: ignore

        # binary {-1,1} to binary {0,1} to int
        # shape: (bsz, n)
        _c = time.time()
        hashes_binary = 0.5 * (1.0 + hashes_binary)
        hashes_int = tf.reduce_sum(tf.cast(hashes_binary, dtype=tf.int64) 
                                   * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(hash_size, tf.int64)),
                                   axis=-1)
        # tensor to array
        hashes = hashes_int.numpy()

        # update hash tables
        _t = time.time()
        for batchidx, seq_hashes in enumerate(hashes):
            hash_index.add(seq_hashes, (fileidx, idx + batchidx))
        table_time += time.time() - _t
        convert_time += _t - _c
        predict_time += _c - _p
        batch_time += _p - _b

    # continue to next iteration
    print('unique hashes:', sum([len(x.counter.keys()) for x in hash_index.hash_tables]))
    print('top 5 collisions:', hash_index.hash_tables[0].counter.most_common(5)) # this one is like fake now
    print('time elapsed:', time.time() - start_time)

#print(hash_index.counter)
print('profiling')
print(batch_time)
print(predict_time)
print(convert_time + table_time) # should be ~1% total so just merge

# write final index and counter to files
outfile1 = os.path.join(results_dir, args.outfile)
with open(outfile1, 'wb') as f:
    pickle.dump(hash_index, f)

print('completed execution. wrote to', outfile1, '.')