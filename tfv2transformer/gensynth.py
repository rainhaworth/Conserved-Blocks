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
        block_lengths_T = np.random.randint(block_min, block_max-1, size=batch_size//2)
        block_lengths_F = np.random.randint(k, block_min-1,  size=batch_size//2)

        indel_counts = np.random.poisson(exp_indel, size=batch_size)
        ins_fracs = np.random.uniform(size=batch_size)

        for idx in range(batch_size):
            # alternate between true and false samples
            label = idx % 2
            labels.append(label)

            # generate first sequence randomly
            if label == 1:
                block_length = block_lengths_T[idx // 2]
                seq1 = gen_seq(np.random.randint(max(block_length, min_len), max_len))
            else:
                block_length = block_lengths_F[idx // 2]
                seq1 = gen_seq(np.random.randint(max(block_length, k), max_len))
            seqs[0].append(seq2kmers(seq1))

            # get conserved "block" as substring of seq1
            if len(seq1) - block_length <= 0:
                block = seq1
            else:
                block_idx = np.random.randint(0, len(seq1)-block_length)
                block = seq1[block_idx:block_idx+block_length]

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
                seqs[1].append(seq2kmers(block))
                continue
            
            # generate random part of seq2
            seq2_rand = gen_seq(np.random.randint(k, max_len-len(block)))

            # choose (count_ins+1) insertion points; model insertion by breaking into fragments
            if count_ins+1 > min(len(seq2_rand), len(block)):
                count_ins = min(len(seq2_rand), len(block))-1

            pos_new_ins = sorted(random.sample(range(len(seq2_rand)), count_ins+1))
            pos_block_ins = sorted(random.sample(range(len(block)), count_ins))
            pos_block_ins.insert(0, 0) # set first insertion point to 0

            # assemble sequence
            seq2 = seq2_rand[0:pos_new_ins[0]]
            for i in range(count_ins):
                seq2 += block[pos_block_ins[i]:pos_block_ins[i+1]]
                seq2 += seq2_rand[pos_new_ins[i]:pos_new_ins[i+1]]
            seq2 += block[pos_block_ins[-1]:]
            seq2 += seq2_rand[pos_new_ins[-1]:]

            seqs[1].append(seq2kmers(seq2))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []