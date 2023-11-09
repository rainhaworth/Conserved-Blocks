# input pipeline for tfv2 original transformer implementation
import os, glob
import numpy as np
import random
#import tfv2transformer.ljqpy as ljqpy

# custom TokenList and pad_to_longest; do not import from dataloader.py
class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>'] + token_list
        self.t2id = {v:k for k,v in enumerate(self.id2t)}
    def id(self, x):    return self.t2id.get(x, 1)
    def token(self, x):    return self.id2t[x]
    def num(self):        return len(self.id2t)
        
def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len)), max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    for i, x in enumerate(xs):
        for j, z in enumerate(x):
            X[i,j] = tokens.id(z)
    return X

# pad to max: always use max_len regardless of inputs
def pad_to_max(xs, tokens, max_len=999):
    X = np.zeros((len(xs), max_len), dtype='int32')
    for i, x in enumerate(xs):
        for j, z in enumerate(x):
            X[i,j] = tokens.id(z)
    return X

# load list of kmers from dna2vec, duplicate, return as itokens, otokens
def LoadKmerDict(dict_file=None, k=8):
    if dict_file is None or os.path.exists(dict_file) == False:
        raise FileNotFoundError(dict_file)
    
    # modified from ljqpy.LoadList()
    with open(dict_file, encoding="utf-8") as f:
        kmers = list(ll[:k] for ll in f.read().split('\n') if len(ll) >= k)
    
    # return as 2 separate TokenLists
    return TokenList(kmers), TokenList(kmers)

# load data from files (as generator) for KmerDataGenerator
# for use with pre-processed data with each sequence in its own .txt file
def LoadFromDir(dir, k=8, max_len=999):
    # get list of filenames
    input_files = glob.glob(os.path.join(dir, "*.txt"))
    random.shuffle(input_files)

    for file in input_files:
        with open(file) as f:
            instr = f.read()
            # skip short/empty strings; hopefully these have been filtered out already
            if len(instr) < k:
                continue
            # yield duplicates
            yield [instr[:max_len+k-1], instr[:max_len+k-1]]

# load fasta data for KmerDataGenerator
def LoadFromDirFasta(dir, k=8, max_len=999):
    # get list of filenames
    input_files = glob.glob(os.path.join(dir, "*.fa"))

    for file in input_files:
        with open(file) as f:
            # loop over all lines
            while True:
                instr = f.readline()
                if not instr:
                    break
                # skip metadata and very short strings
                if instr[0] == '>' or len(instr) < k:
                    continue
                # TODO: handle strings where len > max_len (split w/ redundancies)
                # yield duplicates
                yield [instr[:max_len+k-1], instr[:max_len+k-1]]

# generator: fetch batches of data and write function
def KmerDataGenerator(dir, itokens, otokens, batch_size=64, k=8, max_len=999):
    # split sequences into kmers, store in xs
    Xs = [[], []]
    while True:
        for ss in LoadFromDir(dir, k, max_len):
            for seq, xs in zip(ss, Xs):
                # create list of kmers
                num_kmers = len(seq) - k + 1
                xs.append([seq[i:i+k] for i in range(num_kmers)])
            if len(Xs[0]) >= batch_size:
                # use max instead of longest
                X, Y = pad_to_max(Xs[0], itokens, max_len), pad_to_max(Xs[1], otokens, max_len)
                yield [X, Y], None
                Xs = [[], []]

# very basic generator of pairs of data with shared blocks
# generates padded kmer lists
def gen_simple_contrastive_data(max_len=4096, min_len=500, block_max=4096, block_min=500, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """
    def gen_seq(length):
        return ''.join(random.choice('ACGT') for _ in range(length))
    
    seqs = [[],[]]

    assert batch_size % 2 == 0

    while True:
        # generate (batch_size/2) blocks
        for _ in range(batch_size/2):
            block_length = random.randint(block_min, block_max)
            block = gen_seq(block_length)

            # generate 2 sequences for each block
            for i in range(2):
                # make them at least large enough to hold the block
                len_seq = random.randint(max(block_length, min_len), max_len)
                if len_seq == block_length:
                    num_kmers = len(block) - k + 1
                    seqs[i].append([block[i:i+k] for i in range(num_kmers)])
                seq = gen_seq(len_seq - block_length)
                # insert block at random point
                insert_point = random.randint(0, len(seq)-1)
                seq = seq[:insert_point] + block + seq[insert_point:]
                
                num_kmers = len(seq) - k + 1
                seqs[i].append([seq[i:i+k] for i in range(num_kmers)])

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield np.concatenate([a,b], axis=0)
        seqs = [[],[]]

# build index of file data, stored in memory
# store as sequences to save memory
class DataIndex:
    def __init__(self, dir, itokens, otokens, k=8, max_len=4096, split=False, fasta=False):
        # split: produce separate indices for each file
        # fasta: use LoadFromDirFasta vs LoadFromDir
        self.k = k
        self.itokens = itokens
        self.otokens = otokens
        self.max_len = max_len
        self.split = split

        if fasta:
            loadfn = LoadFromDirFasta
        else:
            loadfn = LoadFromDir

        # TODO: from dir (.txt) vs from fasta
        self.index = []
        if split == False:
            for ss in loadfn(dir, k, max_len):
                # just store X, drop Y
                self.index.append(ss[0])
        else:
            # TODO: write function to avoid this duplicate code
            # grab all files
            input_files = glob.glob(os.path.join(dir, "*.txt"))
            for file in input_files:
                inarr = []
                with open(file) as f:
                    instr = f.read()
                    # skip short/empty strings; hopefully these have been filtered out already
                    if len(instr) < k:
                        continue
                    inarr.append(instr)
                # add array to index
                self.index.append(inarr)

    def get(self, idx, file=None):
        # for indices split across multiple files, pass in the desired file number (array index)
        if file != None:
            data = self.index[file][idx]
        else:
            data = self.index[idx]
        # generate list of kmers, fetch tokens, pad
        num_kmers = len(data) - self.k + 1
        return pad_to_max(
            [data[i:i+self.k] for i in range(num_kmers)],
            self.itokens, self.max_len
        )
    
    def len(self):
        if self.split == False:
            return len(self.index)
        else:
            # iterate over files
            length = 0
            for file in self.index:
                length += len(file)
            return length

if __name__ == '__main__':
    # test code
    itok, otok = LoadKmerDict('../utils/8mers.txt')
    print('loaded', itok.num(), 'tokens')
    gen = KmerDataGenerator('../data-tmp/', itok, otok, batch_size=4, max_len=50)
    i = 0
    for elem in gen:
        print(elem)
        i += 1
        if i > 2:
            break
