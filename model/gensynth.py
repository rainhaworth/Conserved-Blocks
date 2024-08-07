# define data generator
# requires tensorflow, so kept separate from input.py
import numpy as np
import random
from tensorflow import constant, expand_dims, ragged
from .input import pad_to_max, pad_to_min_chunk

# utility functions
def gen_seq(length, lst=False):
    seq_lst = np.random.choice(['A','C','G','T'], size=length)
    if not lst:
        return ''.join(seq_lst)
    return seq_lst
def seq2kmers(seq, k):
    num_kmers = len(seq) - k + 1
    return [seq[i:i+k] for i in range(num_kmers)]

def make_ragged(xs, tokens):
    X = []
    for i, x in enumerate(xs):
        X.append([])
        for j, z in enumerate(x):
            X[i].append(tokens.id(z))
    return ragged.constant(X)

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

    # temporarily added random max_len per batch, sampled from hardcoded list of valid options
    #max_lens = [4500, 7500, 15000]
    max_len_init = max_len
    min_chunk_pop = 0.8 # fraction of chunk that must be populated
    chunksz = 2000 # hardcoded

    # return integer: additional # of nucleotides to generate to meet chunk population bound
    def round_to_pop(seqlen, pop, chunk):
        if seqlen % chunk < chunk * pop and seqlen % chunk != 0:
            return int(np.ceil(chunk * pop) - (seqlen % chunk))
        return 0
        
    while True:
        max_len = np.random.randint(min_len*2, max_len_init)
        block_max = max_len
        # ensure labels are evenly balanced
        sample_labels = np.concatenate([np.ones(batch_size//2), np.zeros(batch_size//2)])
        np.random.shuffle(sample_labels)

        block_lengths = np.random.randint(block_min, block_max-1, size=batch_size)
        for idx in range(batch_size):
            block_length = block_lengths[idx]
            block = gen_seq(block_length)

            if sample_labels[idx] == 1:
                # generate positive labeled sequences
                for i in range(2):
                    # make them at least large enough to hold the block
                    len_seq = random.randint(max(block_length, min_len), max_len-1) # min_len -> max_len-k-block_length

                    # ensure last chunk population bound is satisfied
                    full_len = len_seq + block_length
                    len_seq += round_to_pop(full_len, min_chunk_pop, chunksz)

                    if len_seq == block_length:
                        seqs[i].append(seq2kmers(block, k))
                        continue
                    seq = gen_seq(len_seq - block_length)
                    # insert block at random point
                    if len(seq) < 2:
                        insert_point = 0
                    else:
                        insert_point = random.randint(0, len(seq)-1)
                    seq = seq[:insert_point] + block + seq[insert_point:]
                    seq = seq[:max_len]
                    seqs[i].append(seq2kmers(seq, k))
                labels.append(1)
            else:
                # generate random negative labeled sequences; ignore minimum length
                seq1_len = random.randint(min_len, max_len-1) # for ChunkHash, replaced k -> min_len
                seq2_len = random.randint(min_len, max_len-1)
                seq1_len += round_to_pop(seq1_len, min_chunk_pop, chunksz)
                seq2_len += round_to_pop(seq2_len, min_chunk_pop, chunksz)
                randseq1 = gen_seq(seq1_len) 
                randseq2 = gen_seq(seq2_len)
                seqs[0].append(seq2kmers(randseq1, k))
                seqs[1].append(seq2kmers(randseq2, k))
                labels.append(0)

        # yield complete list of sequences
        #a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        #a, b = make_ragged(seqs[0], tokens), make_ragged(seqs[1], tokens)
        longest_a = len(max(seqs[0], key=len))
        longest_b = len(max(seqs[1], key=len))
        longest = max(longest_a, longest_b)
        a, b = pad_to_min_chunk(seqs[0], tokens, longest, chunksz), pad_to_min_chunk(seqs[1], tokens, longest, chunksz)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []


# adversarial block data
# generates data near the decision boundary
def gen_adversarial_block_data_binary(max_len=4096, min_len=500, block_max=None, block_min=None,
                                      prob_sub=0.05, exp_indel=1, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        prob_sub: probability of random substitution events and insertion/deletion events
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings (not yet implemented)
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """

    boundary_pad = 100
    
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
        # pre-generate random number batches
        block_lengths_T = np.random.randint(block_min, block_max-1, size=batch_size//2)
        block_lengths_F = np.random.randint(k, block_min-boundary_pad,  size=batch_size//2)

        indel_counts = np.random.poisson(exp_indel, size=batch_size)
        ins_fracs = np.random.uniform(size=batch_size)

        for idx in range(batch_size):
            # alternate between true and false samples
            label = idx % 2
            labels.append(label)

            # generate conserved block
            if label == 1:
                block_length = block_lengths_T[idx // 2]
            else:
                block_length = block_lengths_F[idx // 2]
            block = gen_seq(block_length)

            # generate sequences: perturb block then insert into new sequence
            for seqnum in range(2):
                # DELETIONS

                # compute number of insertions and deletions to perform
                count_ins = round(ins_fracs[idx] * indel_counts[idx])
                count_del = indel_counts[idx] - count_ins

                if count_del > 0:
                    # compute total number of nucleotides to delete
                    if label == 0:
                        if count_del >= block_length - k:
                            size_del = block_length - k
                        else:
                            size_del = np.random.randint(count_del, block_length-k)
                    elif block_length - block_min <= count_del:
                        size_del = block_length - block_min
                    else:
                        size_del = np.random.randint(count_del, block_length - block_min)
                    # for each individual deletion event (slightly smaller due to truncation)
                    sizes_del = (np.random.dirichlet(np.ones(count_del)) * size_del).astype(int)

                    # simulate deletion events
                    for dsz in sizes_del:
                        if len(block) - dsz <= 0:
                            block = block[:k]
                            continue
                        pos_del = np.random.randint(0, len(block) - dsz)
                        block = block[:pos_del] + block[pos_del+dsz:]
                
                # SUBSTITUTIONS
                
                # compute number of random substitutions to perform
                count_sub = np.random.binomial(len(block), prob_sub)
                if label == 1:
                    count_sub = min(count_sub, len(block) - block_min)
                else:
                    count_sub = min(count_sub, len(block) - k)

                # simulate substitution events
                if count_sub > 0:
                    seq_sub = gen_seq(count_sub)
                    idx_sub = sorted(random.sample(range(len(block)), count_sub))

                    newblock = block[0:idx_sub[0]]
                    for i in range(count_sub-1):
                        newblock += seq_sub[i]
                        newblock += block[idx_sub[i]+1:idx_sub[i+1]]
                    newblock += seq_sub[-1]
                    if idx_sub[-1]+1 < len(block):
                        newblock += block[idx_sub[-1]+1:]
                    
                    block = newblock

                # ASSEMBLY AND INSERTIONS
                
                # if block is huge, skip all of this
                if max_len - len(block) <= k:
                    seqs[seqnum].append(seq2kmers(block, k))
                    continue
                
                # generate random part of seq
                seq_rand = gen_seq(np.random.randint(k, max_len-len(block)))

                # choose (count_ins+1) insertion points; model insertion by breaking into fragments
                if count_ins+1 > min(len(seq_rand), len(block)):
                    count_ins = min(len(seq_rand), len(block))-1

                rand_ins_idxs = sorted(random.sample(range(len(seq_rand)), count_ins+1))
                pos_block_ins = sorted(random.sample(range(len(block)), count_ins))
                pos_block_ins.insert(0, 0) # set first insertion point to 0

                # assemble sequence
                seq = seq_rand[0:rand_ins_idxs[0]]
                for i in range(count_ins):
                    seq += block[pos_block_ins[i]:pos_block_ins[i+1]]
                    seq += seq_rand[rand_ins_idxs[i]:rand_ins_idxs[i+1]]
                seq += block[pos_block_ins[-1]:]
                seq += seq_rand[rand_ins_idxs[-1]:]

                seqs[seqnum].append(seq2kmers(seq, k))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []


# alt training regime: adversarial single chunks
def gen_adversarial_chunks_binary(chunk_size=1024, min_pop=0.8, min_shared=512, boundary_pad=50,
                                  prob_sub=0.05, exp_indel=1, batch_size=32, tokens=None, k=4):
    seqs = [[],[]]
    labels = []

    # leaving indels unimplemented for now to simplify

    while True:
        # pre-generate random number batches
        shared_T = np.random.randint(min_shared, chunk_size-1, size=batch_size//2)
        shared_F = np.random.randint(k, min_shared-boundary_pad, size=batch_size//2)

        #indel_counts = np.random.poisson(exp_indel, size=batch_size)
        #ins_fracs = np.random.uniform(size=batch_size)

        for idx in range(batch_size):
            # alternate between true and false samples
            label = idx % 2
            labels.append(label)

            # generate conserved region
            if label == 1:
                shared_length = shared_T[idx // 2]
            else:
                shared_length = shared_F[idx // 2]
            shared = list(gen_seq(shared_length, lst=True))

            #print('max:', max(shared_F), max(shared_T))

            # generate sequences: perturb then insert into new sequence
            for seqnum in range(2):
                # DELETIONS
                """
                # compute number of insertions and deletions to perform
                count_ins = round(ins_fracs[idx] * indel_counts[idx])
                count_del = indel_counts[idx] - count_ins

                if count_del > 0:
                    # compute total number of nucleotides to delete
                    if label == 0:
                        if count_del >= block_length - k:
                            size_del = block_length - k
                        else:
                            size_del = np.random.randint(count_del, block_length-k)
                    elif block_length - block_min <= count_del:
                        size_del = block_length - block_min
                    else:
                        size_del = np.random.randint(count_del, block_length - block_min)
                    # for each individual deletion event (slightly smaller due to truncation)
                    sizes_del = (np.random.dirichlet(np.ones(count_del)) * size_del).astype(int)

                    # simulate deletion events
                    for dsz in sizes_del:
                        if len(block) - dsz <= 0:
                            block = block[:k]
                            continue
                        pos_del = np.random.randint(0, len(block) - dsz)
                        block = block[:pos_del] + block[pos_del+dsz:]
                """
                # SUBSTITUTIONS
                
                # compute number of random substitutions to perform
                count_sub = np.random.binomial(len(shared), prob_sub)
                if label == 1:
                    count_sub = min(count_sub, len(shared) - min_shared)
                else:
                    count_sub = min(count_sub, k)

                # simulate substitution events
                if count_sub > 0:
                    seq_sub = gen_seq(count_sub)
                    idx_sub = random.sample(range(len(shared)), count_sub)

                    for j, i in enumerate(idx_sub):
                        shared[i] = seq_sub[j]

                # ASSEMBLY AND INSERTIONS
                
                # if shared region is huge, skip all of this and use the sequence as is
                if chunk_size - len(shared) <= k:
                    seq = ''.join(shared)
                    seqs[seqnum].append(seq2kmers(seq, k))
                    continue
                
                # generate rest of sequence as new random DNA
                # TODO: bias toward full chunks
                max_seq_len = max(chunk_size + k - len(shared) - 1, 0)
                min_seq_len = max(int(chunk_size*min_pop) + k - len(shared) - 1, 0)
                if min_seq_len >= max_seq_len:
                    seq_len = max_seq_len
                else:
                    seq_len = np.random.randint(min_seq_len, max_seq_len)
                #print(min_seq_len, max_seq_len, ';', len(shared))
                seq = list(gen_seq(seq_len, lst=True))

                count_ins = 0

                # choose (count_ins+1) insertion points; model insertion by breaking into fragments
                # ensure we aren't doing too many insertions; should be unnecessary in most cases
                if count_ins+1 > min(len(seq), len(shared)):
                    count_ins = min(len(seq), len(shared))-1

                # get insertion positions
                seq_ins_idxs = sorted(random.sample(range(len(seq)), count_ins+1))
                if count_ins > 0:
                    shared_ins_idxs = sorted(random.sample(range(len(shared)), count_ins))
                    shared_ins_idxs.insert(0, 0) # set first insertion point to 0
                else:
                    shared_ins_idxs = [0]

                # assemble sequence
                for i in range(count_ins+1):
                    r_ins = seq_ins_idxs[i]
                    s_cur = shared_ins_idxs[i]
                    if i+1 < len(shared_ins_idxs):
                        s_next = shared_ins_idxs[i+1] # slice to next subsection
                    else:
                        s_next = None # slice to end of list
                    # this is the best way to insert multiple elements in native python and i'm not happy about it
                    seq[r_ins:r_ins] = shared[s_cur:s_next]

                # convert to string
                seq = ''.join(seq)

                seqs[seqnum].append(seq2kmers(seq, k))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, chunk_size), pad_to_max(seqs[1], tokens, chunk_size)
        # force overfitting
        for _ in range(1):
            yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []