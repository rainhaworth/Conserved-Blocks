# kmer graph based encoder
import tensorflow as tf
import numpy as np

from .chunk_hash import bucket_metric_factory

# helper function: less gross looking conv2d
def conv2D(dim, kernel=1, stride=1):
    return tf.keras.layers.Conv2D(dim, kernel, stride, 'same')

# helper function: convert batch of sequences to batch of k^4 x k^4 kmer graphs w/ fixed separation
def seqs2graphs(seqs, sep=1, k=4):
    # adjust for 0 = pad
    seqs -= 1
    # get batch size, graph size, sequence length
    sh = tf.shape(seqs)
    bsz = sh[0]
    gsz = k ** 4
    seqlen = sh[-1]
    # make tensor of all edges + batch indices
    batch_idxs = tf.tile(tf.range(0,bsz)[:,None], [1,seqlen-sep])
    seqs_A = seqs[:,:seqlen-sep]
    seqs_B = seqs[:,sep:]
    edges = tf.stack([batch_idxs, seqs_A, seqs_B], axis=-1)
    # make graphs, adding 1 for each edge
    updates = tf.ones_like(seqs_A)
    return tf.scatter_nd(edges, updates, (bsz, gsz, gsz))

class ResBlock2D(tf.keras.layers.Layer):
    def __init__(self, dim=128, kernelsz=3, layers=2, layernorm=True, project=False):
        super(ResBlock2D, self).__init__()
        self.layernorm = None
        self.project = None
        self.convs = []
        self.relu = tf.keras.layers.ReLU()
        # dimension increase version; assume dim = previous dim * 2
        if project:
            self.project = conv2D(dim, 1, 2)
            self.convs.append(conv2D(dim, kernelsz, 2))
            layers -= 1
        # create convolutional layers
        self.convs += [conv2D(dim, kernelsz) for _ in range(layers)]
        if layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization()
            
    def call(self, x):
        x_init = x
        if self.project is not None:
            x_init = self.project(x_init)
        # convs, intermediate relus
        for conv in self.convs[:-1]:
            x = conv(x)
            if self.layernorm is not None:
                x = self.layernorm(x)
            x = self.relu(x)
        # last conv
        x = self.convs[-1](x)
        # skip + relu
        x = x + x_init
        return self.relu(x)

class GraphHash:
    def __init__(self, tokens, k=4, chunksz=1024, d_model=256, hashsz=64, n_hash=32):
        self.tokens = tokens
        self.d_model = d_model
        self.c = chunksz
        self.h = hashsz
        self.n = n_hash
        self.k = k
        d = d_model

        # resnet-34 encoder
        self.enc_layers = [conv2D(d, 7, 2), tf.keras.layers.LayerNormalization(), tf.keras.layers.ReLU(),
                           tf.keras.layers.MaxPool2D((3,3), 2, padding='same'),
                           ResBlock2D(d), ResBlock2D(d), ResBlock2D(d),
                           ResBlock2D(d*2, project=True), ResBlock2D(d*2), ResBlock2D(d*2), ResBlock2D(d*2),
                           ResBlock2D(d*4, project=True), ResBlock2D(d*4), ResBlock2D(d*4), ResBlock2D(d*4), ResBlock2D(d*4), ResBlock2D(d*4),
                           ResBlock2D(d*8, project=True), ResBlock2D(d*8), ResBlock2D(d*8),
                           #tf.keras.layers.GlobalAveragePooling2D()
                           ]
        
        # hasher
        self.flatten = tf.keras.layers.Flatten()
        self.hash_layer = tf.keras.layers.Dense(hashsz*n_hash, activation='tanh')

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        # construct graphs
        inseqs = [inseqs1, inseqs2]
        inseqs_concat = tf.concat(inseqs, 0)
        graphs_list = [seqs2graphs(inseqs_concat, i, self.k) for i in range(1,4)]
        graphs = tf.stack(graphs_list, axis=-1)
        graphs = tf.cast(graphs, tf.float32)
        graphs1, graphs2 = tf.split(graphs, 2, axis=0)

        # encode and hash
        hashes = []
        for x in (graphs1, graphs2):
            for layer in self.enc_layers:
                x = layer(x)
            x = tf.math.l2_normalize(x, axis=-1) # for some reason this continues to be critical to the model not exploding
            x = self.flatten(x)
            x = self.hash_layer(x)
            x = tf.reshape(x, [-1, self.n, self.h])
            hashes.append(x)

        hash_loss, hash_precision, hash_recall, unique = bucket_metric_factory(self.n, self.h)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall, unique]
                           )