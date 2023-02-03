# Reusable helper functions for impementing pipelines

import numpy as np
import tensorflow as tf
from dna2vec.dna2vec.multi_k_model import MultiKModel

# TODO: make k-mer embedding into a tf.keras.layers.Layer

# Generate a MultiKModel
def init_dna2vec():
    filepath = 'pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
    return MultiKModel(filepath)

# K-mer embedding from config string
def contig_to_kmers(contigstr, mk_model):
    length = len(contigstr) - 8 + 1
    embed = tf.zeros([length, 100])
    for i in range(length):
        embed[i] = mk_model.vector(contigstr[i:i+8])
    return embed

# TODO: vector-based clustering
