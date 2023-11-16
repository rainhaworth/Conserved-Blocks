# imported from cluster-tfv2-dbscan
import os, sys, random
import tfv2transformer.input as dd
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf

# set max length
#max_len = 4096
#max_len = 512
max_len = 2048
k = 4

itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt')
#gen = dd.KmerDataGenerator('./data-tmp/', itokens, otokens, batch_size=1, max_len=max_len)

print('tokens:', itokens.num(), otokens.num())

# (don't need this for now)
# make indexed list of all sequences
seqs = dd.DataIndex('/fs/nexus-scratch/rhaworth/synth/', itokens, otokens, max_len=max_len, fasta=True)

from tfv2transformer.transformer import LRSchedulerPerStep
from tfv2transformer.contrastive_encoder import ContrastiveEncoder


#d_model=256
d_model = 512
block_size = 64
#s2s = Transformer(itokens, otokens, len_limit=512, d_model=d_model, d_inner_hid=512, \
#                   n_head=8, layers=2, length=max_len, block_size=block_size, dropout=0.1)
#s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
#                   n_head=8, layers=2, dropout=0.1)
s2s = ContrastiveEncoder(itokens, len_limit=1024, d_model=d_model, d_inner_hid=512, \
                   n_head=8, layers=2, length=max_len, block_size=block_size, dropout=0.1)


mfile = '/fs/nexus-scratch/rhaworth/models/contrastive.model.h5'

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try: s2s.model.load_weights(mfile)
except: print('No model file found at', mfile)

# make encoder-only model
s2s.make_encode_model()

# define distance metrics
def cos_loss_dist(x, y):
    cos_loss = -tf.reduce_sum(tf.tensordot(tf.math.l2_normalize(tf.squeeze(x)), tf.math.l2_normalize(tf.squeeze(y)), 1))
    return cos_loss

def l1_dist(x, y):
    return tf.reduce_sum(x - y)

def l2_dist(x,y):
    return tf.norm(x - y, 2)

def linf_dist(x, y):
    return tf.norm(x - y, np.inf)

def fro_dist(x, y):
    return tf.norm(x-y)

# test metric
def diff_count(x, y):
    eq = tf.math.equal(x, y)
    y, _, count = tf.unique_with_counts(tf.reshape(eq, [-1]))
    return [y,count]

def l1_dist(x, y):
    return tf.reduce_sum(x - y)

def l2_dist(x,y):
    return tf.norm(x - y, 2)

# choose metric for tests
metric = cos_loss_dist

# TODO: design unit tests
    # 1. identical strings [done]
    # 2. nearly-identical strings
    # 3. strings with a large shared block, same position [done]
    # 4. "", different positions [done]
    # 5. strings with sparser/non-identical shared regions, same pos
    # 6. "", different pos
    # 7. random strings
    # 8. strings with no shared information
    # ideally, we should see a rough gradient from 100% to 0% similarity (or 0 to max dist)
    # also, 3-4 and 5-6 should be as similar to each other as possible
    # bonus goal: it shouldn't matter whether the non-block DNA is random (minimally similar) or has no shared content at all

# import string generation, padding
from utils.synthdata import gen_seq
from tfv2transformer.input import pad_to_max

# convert sequence to kmers, with padding
def seq2kmers(seq, k=8):
    # new implementation
    num_kmers = len(seq) - k + 1
    kmers = np.zeros((1, max_len))
    for i in range(num_kmers):
        kmers[0,i] = itokens.id(seq[i:i+k])
    return kmers

# first random seq
randseq1 = gen_seq(max_len)
pred_1 = s2s.encode_model(seq2kmers(randseq1, k))

# test 1: identical strings/embeddings
print("test 1 dist:", metric(pred_1, pred_1))

# TODO: test 2

# test 3: identical blocks in same position, different strings
#block_len = 1000
#pos = 450
block_len = 150
pos = 100
block1 = gen_seq(block_len)
randseq2 = gen_seq(max_len)

randseq1_block1 = randseq1[:pos] + block1 + randseq1[pos+block_len:]
randseq2_block1 = randseq2[:pos] + block1 + randseq2[pos+block_len:]

pred_test3_1 = s2s.encode_model(seq2kmers(randseq1_block1, k))
pred_test3_2 = s2s.encode_model(seq2kmers(randseq2_block1, k))

print("test 3 dist:", metric(pred_test3_1, pred_test3_2))

# test 4: same blocks, different position
pos2 = 300
randseq2_block1_pos2 = randseq2[:pos2] + block1 + randseq2[pos2+block_len:]
pred_test4 = s2s.encode_model(seq2kmers(randseq2_block1_pos2, k))

print("test 4 dist:", metric(pred_test3_1, pred_test4))

# TODO: test 5 and 6

# test 7: random strings
pred_test7 = s2s.encode_model(seq2kmers(randseq2, k))
print("test 7 dist:", metric(pred_1, pred_test7))

# TODO: test 8
