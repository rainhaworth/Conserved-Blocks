# input pipeline for tfv2 original transformer implementation
import os, glob
import numpy as np
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
            instr = f.read()
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
        for ss in LoadFromDirFasta(dir, k, max_len):
            for seq, xs in zip(ss, Xs):
                # create list of kmers
                num_kmers = len(seq) - k + 1
                xs.append([seq[i:i+k] for i in range(num_kmers)])
            if len(Xs[0]) >= batch_size:
                X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
                yield [X, Y], None
                Xs = [[], []]

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