# hash all sequences from .fasta files in directory
import os
import pickle
import model.input as dd
from tqdm.auto import tqdm
import time
import argparse
import tensorflow as tf

from model.input import pad_to_min_chunk, HashIndex

# command line args
parser = argparse.ArgumentParser()
parser.add_argument('--chunk_size', default=2000, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--hash_size', default=64, type=int)
parser.add_argument('--chunk_pop', default=0.8, type=float) # fraction of chunk that must be populated
args = parser.parse_args()

# set global vars
chunk_size = args.chunk_size
k = args.k
d_model = args.d_model
hash_size = args.hash_size
chunk_pop = args.chunk_pop

# load kmer dict
itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)

# initialize model
from model.chunk_hash import DiscretizedChunkHash
ch = DiscretizedChunkHash(itokens, d_model=d_model, hashsz=hash_size, chunksz=chunk_size)

# load weights
# TODO: make this a parameter
mfile = '/fs/nexus-scratch/rhaworth/models/chunkhash.model.h5'
ch.compile()
try: ch.model.load_weights(mfile)
except: print('\nno model found')
ch.make_inference_model()

# utility functions
def maxlen_to_batch(maxlen, vram=24, res_layers=6):
    # vram in GB; TODO: fetch via nvidia-smi or tf.config.experimental.get_memory_info
    # assume padded to min chunk
    vram = vram * 1e9
    copy_mult = 6 + res_layers
    # also divide by float size (4 bytes)
    capacity = vram // (4 * d_model * copy_mult)
    # compute batch size
    return capacity // maxlen

### HASHING ###

# TODO: these should be parameters too
dataset_path = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
results_dir = '/fs/nexus-scratch/rhaworth/output/'
ext = 'fa'

hash_index = HashIndex(dataset_path, ext)
print('files found:', len(hash_index.filenames))

# set start time
start_time = time.time()

for fileidx, filename in enumerate(hash_index.filenames):
    print('file', fileidx, ':', filename)

    # read file, get sequences, sort by length
    seqs = hash_index.seqs_from_file(fileidx)
    
    """
    # prune short sequences
    seq_lens = [len(s) for s in seqs]
    i = 0
    # we could binary search but do this later
    while seq_lens[i] < chunk_pop * chunk_size:
        i += 1
    seqs = seqs[i:]"""

    # enter loop until sequence list exahusted
    idx = len(seqs)-1
    pbar = tqdm(total=idx)
    while idx > 0:
        # get batch size, update idx
        maxlen = len(seqs[idx])
        bsz = maxlen_to_batch(maxlen)
        #print(maxlen, bsz)

        bsz = int(min(bsz, idx))

        # construct batch
        batch = seqs[idx-bsz:idx+1]

        batch_tokens = pad_to_min_chunk(batch, itokens, maxlen, chunk_size)
        
        # call model to get hash
        hashes_binary = ch.hasher.predict(batch_tokens, verbose=0) # type: ignore

        # binary (-1,1) to binary (0,1) to int
        hashes_binary = 0.5 * (1.0 + hashes_binary)
        hashes_int = tf.reduce_sum(tf.cast(hashes_binary, dtype=tf.int64) 
                                   * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(hash_size, tf.int64)),
                                   axis=-1)
        # tensor to array
        hashes = hashes_int.numpy()

        # update index + counter
        for seqidx, seqhashes in enumerate(hashes):
            for chunkidx, hash in enumerate(seqhashes):
                # drop invalid hashes
                if hash == 0:
                    continue
                # TODO: convert to short
                # convert seqidx from index in batch to index in seqs array
                hash_index.add(hash, fileidx, seqidx + idx - bsz, chunkidx)

        # update loop index + tqdm
        idx -= bsz
        pbar.update(bsz)

        # everything gets converted to float for some reason so fix that
        idx = int(idx)
        bsz = int(bsz)

    pbar.close()

    # continue to next iteration
    print('unique hashes:', len(hash_index.counter.keys()))
    print('top 5 collisions:', hash_index.counter.most_common(5))
    print('time elapsed:', time.time() - start_time)

print(hash_index.counter)

# write final index and counter to files
outfile1 = os.path.join(results_dir, 'hashindex.pickle')
with open(outfile1, 'wb') as f:
    pickle.dump(hash_index, f)

print('completed execution. wrote to', outfile1, '.')