# input pipeline for tfv2 original transformer implementation
import os, glob
import numpy as np
import random
#import tfv2transformer.ljqpy as ljqpy
from tensorflow import constant, expand_dims

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
            yield [instr[:max_len+k-1], instr[:max_len+k-1]], None # no metadata

# load fasta data for KmerDataGenerator
def LoadFromDirFasta(dir, k=8, max_len=999):
    # get list of filenames
    input_files = glob.glob(os.path.join(dir, "*.fa"))

    for file in input_files:
        with open(file) as f:
            # loop over all lines
            metadata = ''
            while True:
                instr = f.readline()
                if not instr:
                    break
                # grab metadata
                if instr[0] == '>' or len(instr) < k:
                    metadata = instr[1:]
                    continue
                # TODO: handle strings where len > max_len (split w/ redundancies)
                # yield duplicates
                yield [instr[:max_len+k-1], instr[:max_len+k-1]], metadata
                metadata = ''

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
def gen_simple_contrastive_data(max_len=4096, min_len=500, block_max=None, block_min=None, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings (not yet implemented)
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """
    def gen_seq(length):
        return ''.join(np.random.choice(['A','C','G','T'], size=length))
    
    seqs = [[],[]]

    assert batch_size % 2 == 0

    # set parameters
    if block_max is None or block_max > max_len:
        block_max = max_len
    if block_min is None:
        block_min = min_len
    elif block_min > max_len:
        block_min = max_len

    while True:
        # generate (batch_size/2) blocks
        for _ in range(batch_size//2):
            block_length = random.randint(block_min, block_max)
            block = gen_seq(block_length)

            # generate 2 sequences for each block
            for i in range(2):
                # make them at least large enough to hold the block
                len_seq = random.randint(max(block_length, min_len), max_len)
                if len_seq == block_length:
                    num_kmers = len(block) - k + 1
                    seqs[i].append([block[i:i+k] for i in range(num_kmers)])
                    continue
                seq = gen_seq(len_seq - block_length)
                # insert block at random point
                if len(seq) < 2:
                    insert_point = 0
                else:
                    insert_point = random.randint(0, len(seq)-1)
                seq = seq[:insert_point] + block + seq[insert_point:]
                
                num_kmers = len(seq) - k + 1
                seqs[i].append([seq[i:i+k] for i in range(num_kmers)])

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield np.concatenate([a,b], axis=0)
        seqs = [[],[]]

# very basic generator of pairs of data with shared blocks
# generates padded kmer lists
# for binary classification
def gen_simple_block_data_binary(max_len=4096, min_len=500, block_max=None, block_min=None, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings (not yet implemented)
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """
    def gen_seq(length):
        return ''.join(np.random.choice(['A','C','G','T'], size=length))
    def seq2kmers(seq):
        num_kmers = len(seq) - k + 1
        return [seq[i:i+k] for i in range(num_kmers)]
    
    seqs = [[],[]]
    labels = []

    assert batch_size % 2 == 0

    # set parameters
    if block_max is None or block_max > max_len:
        block_max = max_len
    if block_min is None:
        block_min = min_len
    elif block_min > max_len:
        block_min = max_len

    while True:
        sample_labels = np.random.randint(2, size=batch_size)
        block_lengths = np.random.randint(block_min, block_max-1, size=batch_size)
        for idx in range(batch_size):
            block_length = block_lengths[idx]
            block = gen_seq(block_length)

            if sample_labels[idx] == 1:
                # generate positive labeled sequences
                for i in range(2):
                    # make them at least large enough to hold the block
                    len_seq = random.randint(max(block_length, min_len), max_len-1)
                    if len_seq == block_length:
                        seqs[i].append(seq2kmers(block))
                        continue
                    seq = gen_seq(len_seq - block_length)
                    # insert block at random point
                    if len(seq) < 2:
                        insert_point = 0
                    else:
                        insert_point = random.randint(0, len(seq)-1)
                    seq = seq[:insert_point] + block + seq[insert_point:]
                    seq = seq[:max_len]
                    seqs[i].append(seq2kmers(seq))
                    # TODO: perturb block and randomly add indels after first iteration
                labels.append(1)
            else:
                # generate random negative labeled sequences; ignore minimum length
                randseq1 = gen_seq(random.randint(k, max_len-1))
                randseq2 = gen_seq(random.randint(k, max_len-1))
                seqs[0].append(seq2kmers(randseq1))
                seqs[1].append(seq2kmers(randseq2))
                labels.append(0)

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []

# build index of file data, stored in memory
# store as sequences to save memory
class DataIndex:
    def __init__(self, dir, itokens, otokens, k=8, max_len=4096, split=False, fasta=False, metadata=False):
        # split: produce separate indices for each file
        # fasta: use LoadFromDirFasta vs LoadFromDir
        self.k = k
        self.itokens = itokens
        self.otokens = otokens
        self.max_len = max_len
        self.split = split
        self.mdindex = None

        if fasta:
            loadfn = LoadFromDirFasta
        else:
            loadfn = LoadFromDir

        if metadata:
            self.mdindex = []
        # TODO: from dir (.txt) vs from fasta
        self.index = []
        if split == False:
            for ss, md in loadfn(dir, k, max_len):
                # just store X, drop Y
                self.index.append(ss[0])
                if metadata and md is not None:
                    self.mdindex.append(md)
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
            [[data[i:i+self.k] for i in range(num_kmers)]],
            self.itokens, self.max_len
        )
    
    def getmd(self, idx):
        if self.mdindex is None:
            return None
        return self.mdindex[idx]

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
