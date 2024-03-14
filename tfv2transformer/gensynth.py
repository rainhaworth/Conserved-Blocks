# define data generator
# requires tensorflow, so kept separate from input.py
import numpy as np
import random
from tensorflow import constant, expand_dims
from .input import pad_to_max

# generate pairs of data with shared blocks
# output = list of padded kmer lists
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