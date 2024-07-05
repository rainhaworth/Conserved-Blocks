# evaluate hash clusters
from model.input import HashIndex
import pickle
import argparse
from Bio import pairwise2

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_size', default=2000, type=int)
parser.add_argument('--hash_size', default=64, type=int)
parser.add_argument('--chunk_pop', default=0.8, type=float) # fraction of chunk that must be populated
parser.add_argument('--index_fn', default='/fs/nexus-scratch/rhaworth/output/hashindex.pickle', type=str)
args = parser.parse_args()

# set global vars
chunk_size = args.chunk_size
hash_size = args.hash_size
chunk_pop = args.chunk_pop
index_fn = args.index_fn

# get index
with open(index_fn, 'rb') as f:
    hash_index = pickle.load(f)
    assert isinstance(hash_index, HashIndex)

# iterate over hashes
file_curr = -1
for hash in hash_index.counter.keys():
    # for now just do pairwise alignment of 2-element clusters, skip others
    count = hash_index.counter[hash]
    if count != 2:
        continue
    # get location, extract sequences then chunks
    locs = hash_index.index[hash]
    chunks = []
    for loc in locs:
        filenum, seqnum, chunknum = loc
        if file_curr != filenum:
            seqs = hash_index.seqs_from_file(filenum)
            file_curr = filenum

        seq = seqs[seqnum]
        chunk = seq[chunknum*chunk_size : (chunknum+1)*chunk_size]

        chunks.append(chunk)
    
    # if we have an empty chunk for some reason, skip
    # this seems to be happening even though it shouldn't be possible...
    if min([len(c) for c in chunks]) == 0:
        continue

    print('hash:', hash)

    # get best alignment
    align = pairwise2.align.globalms(chunks[0], chunks[1], 2, -1, -3, -.1)
    print(pairwise2.format_alignment(*align[0]))