import numpy as np
import random
import argparse

#np.random.seed(0)

# random sequence generator
def gen_seq(length, lst=False):
    seq_lst = np.random.choice(['A','C','G','T'], size=length)
    if not lst:
        return ''.join(seq_lst)
    return seq_lst

# generate n pairs of identical-length sequences following evolution
def sim_evo_fasta(seqlen=1024, n=10, prob_sub=0.05, exp_indel_count=1.0, exp_indel_size=5.0):
    seqs = [[],[]]

    # indel sampling; fixed for each pair
    indel_counts = np.random.poisson(exp_indel_count, size=n)
    ins_fracs = np.random.uniform(size=n)
    indel_sizes = [np.random.poisson(exp_indel_size, size=ic) for ic in indel_counts]

    ins_counts = np.int32(np.round(indel_counts * ins_fracs))
    #del_counts = indel_counts - ins_counts

    for idx in range(n):
        # generate intial sequence
        seq = gen_seq(seqlen)

        # simulate evolution on both sequences
        for seqnum in range(2):
            # DELETIONS

            # insertions are listed first in indel_sizes, so skip them
            for i in range(ins_counts[idx], indel_counts[idx]):
                # get size
                dsz = indel_sizes[idx][i]
                
                # if we would delete more than the full sequence, proceed with empty sequence
                if len(seq) - dsz <= 0:
                    seq = ''
                    break

                # delete random chunk
                pos_del = np.random.randint(0, len(seq)-dsz)
                seq = seq[:pos_del] + seq[pos_del+dsz:]
            
            # SUBSTITUTIONS
            
            if len(seq) > 0:
                seq = list(seq)
                # compute number of random substitutions to perform
                count_sub = np.random.binomial(len(seq), prob_sub)

                # simulate substitution events
                if count_sub > 0:
                    seq_sub = gen_seq(count_sub)
                    idx_sub = random.sample(range(len(seq)), count_sub)

                    for j, i in enumerate(idx_sub):
                        seq[i] = seq_sub[j]
            else:
                # guarantee seq is a list
                seq = []

            # INSERTIONS

            for i in range(ins_counts[idx]):
                # generate new sequence and insert at random position
                isz = indel_sizes[idx][i]
                ins_seq = list(gen_seq(isz, lst=True))
                ins_pos = np.random.randint(0, len(seq))
                
                # unfortunately this is the best way to insert multiple list elements in python
                seq[ins_pos:ins_pos] = ins_seq

            # convert to string
            seq = ''.join(seq)

            # enforce length; add or remove DNA if necessary
            if len(seq) > seqlen:
                # pick random position to trim
                trim_start = np.random.randint(0, len(seq) - seqlen)
                seq = seq[trim_start : trim_start+seqlen]

            elif len(seq) < seqlen:
                # randomly distribute padding on either side of seq
                pad = gen_seq(seqlen - len(seq))
                pad_break = np.random.randint(0, len(pad))
                seq = pad[:pad_break] + seq + pad[pad_break:]

            seqs[seqnum].append(seq)

    return seqs

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--seqlen', default=2000, type=int)
    parser.add_argument('-n', '--num', default=10, type=int)
    parser.add_argument('-p', '--prob_sub', default=0.05, type=float) # fraction of chunk that must be populated
    parser.add_argument('-c', '--exp_indel_count', default=1.0, type=float)
    parser.add_argument('-s', '--exp_indel_size', default=5.0, type=float)
    args = parser.parse_args()

    seqs = sim_evo_fasta(args.seqlen, args.num, args.prob_sub, args.exp_indel_count, args.exp_indel_size)

    # print in fasta format; pipe output to desired file
    for i in range(args.num):
        for j in range(2):
            print('>seq' + str(j))
            #print('>' + str(i) + ',' + str(j))
            print(seqs[j][i])