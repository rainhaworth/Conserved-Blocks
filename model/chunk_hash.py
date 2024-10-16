# learning to hash chunks of sequences such that similar sequences tend to produce collisions
# some sections modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf
import numpy as np
from model.hash_metrics import *

# COMPONENTS

# reduce entire chunk to 1 feature map
class ChunkReduceBlock(tf.keras.layers.Layer):
    def __init__(self, chunksz=1024, dim=128, upscale=1):
        super(ChunkReduceBlock, self).__init__()
        # assemble convs to perform reduction
        # assume chunksz can be mostly factorized by these, i.e. the largest prime factor is small
        factors = [2,3,4,5]
        sz_curr = chunksz
        dim_curr = dim
        self.redconvs = []
        for factor in factors[::-1]:
            while sz_curr % factor == 0:
                # maybe these don't need to have a bigger filter than the stride, test later
                sz_curr = sz_curr // factor
                dim_curr *= upscale
                self.redconvs.append(tf.keras.layers.Conv1D(dim_curr, factor, factor,
                                                            padding='valid', activation='tanh'))
        # extra conv if needed
        if sz_curr > 1:
            self.redconvs.append(tf.keras.layers.Conv1D(dim_curr*upscale, sz_curr, sz_curr,
                                                        padding='valid', activation='tanh'))
        # resblocks; same scale as previous redconv layer
        self.resblocks = [ResBlock1D(dim=dim*(upscale**i)) for i in range(len(self.redconvs))]
    def call(self, x):
        for res, red in zip(self.resblocks, self.redconvs):
            x = res(x)
            x = red(x)
        return x

# reduce continuous sequence with overlapping convs
class ConvReduceBlock(tf.keras.layers.Layer):
    def __init__(self, downsample=1024, layers=4, dim=128):
        super(ConvReduceBlock, self).__init__()
        # reduce 'downsample' kmers to 1 across N layers
        self.ds_per_layer = int(downsample ** (1 / layers))
        self.redconvs = [tf.keras.layers.Conv1D(dim, self.ds_per_layer, self.ds_per_layer, padding='valid', activation='tanh') for _ in range(layers)]
        # extra conv; reduce if needed, otherwise use simple 3-wide conv
        ds_current = self.ds_per_layer ** layers
        ds_last = downsample // ds_current
        self.ds_total = ds_current*ds_last
        self.redconvs.append(tf.keras.layers.Conv1D(dim, ds_last, ds_last, padding='valid', activation='tanh'))
        # resblocks
        self.resblocks = [ResBlock1D(dim=dim, layers=4) for _ in range(len(self.redconvs))]
    def call(self, x):
        for res, red in zip(self.resblocks, self.redconvs):
            x = res(x)
            x = red(x)
        return x

# 1D residual block
class ResBlock1D(tf.keras.layers.Layer):
    def __init__(self, dim=128, kernelsz=3, layers=2, layernorm=False, upsample=False):
        super(ResBlock1D, self).__init__()
        self.layernorm = None
        self.upsample = None
        # create convolutional layers
        self.convs = [tf.keras.layers.Conv1D(dim, kernelsz, padding='same') for _ in range(layers)]
        if layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()
        if upsample:
            self.upsample = tf.keras.layers.Dense(dim, use_bias=False)
    def call(self, x):
        x_init = x
        if self.upsample is not None:
            x_init = self.upsample(x_init)
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

# inception layer w/ dim reduction
# hybrid of original 2014 architecture and LSB architecture
class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, dim=128, maxlen=7, pool=2, layernorm=False, reduce=True):
        super(InceptionLayer, self).__init__()

        self.dim = dim
        if reduce:
            # require divisibility by # of filters
            assert (dim*pool) % (maxlen+1) == 0
            self.dim = (dim*pool) // (maxlen+1)

        self.filters = [tf.keras.layers.Conv1D(self.dim, f+1, activation='relu', padding='same')
                        for f in range(maxlen)]
        self.filters.append(tf.keras.layers.MaxPool1D(2, 1, 'same'))
        self.pool = tf.keras.layers.MaxPool1D(pool)
        #self.norm = tf.keras.layers.LayerNormalization()

        self.reducers = [None for _ in range(len(self.filters))]
        if reduce:
            # first layer does not need a reducer
            self.reducers[1:] = [tf.keras.layers.Conv1D(self.dim, 1, activation='relu', padding='same')
                                 for _ in range(len(self.filters)-1)]
    
    def call(self, x):
        # apply filters
        out = []
        for reducer, filter in zip(self.reducers, self.filters):
            o = x
            if reducer != None:
                o = reducer(o)
            o = filter(o)
            #o = self.pool(o)
            out.append(o)
        # concat feature maps
        return tf.concat(out, axis=-1) # removed pool

# MLP for hashing
class MLPHasher(tf.keras.layers.Layer):
    def __init__(self, hashdim=64, layers=4):
        super(MLPHasher, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(hashdim, activation='relu')]
        self.hasher = tf.keras.layers.Dense(hashdim, activation='tanh')
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.hasher(x)
        return x

# hash layer that splits the input (b x s x c) feature map before hashing
class SplitHasher(tf.keras.layers.Layer):
    def __init__(self, hashdim=64, n_hash=8, indim=256):
        super(SplitHasher, self).__init__()
        assert indim % n_hash == 0
        # redistributes channel information among hash functions
        self.channel_remap = tf.keras.layers.Dense(indim, use_bias=False)
        # reshapes to (b x s x n x c/n)
        self.reshape_1 = tf.keras.layers.Reshape((-1, n_hash, indim // n_hash))
        # permutes to (b x n x s x c/n)
        self.permuter = tf.keras.layers.Permute((2,1,3))
        # reshapes to (b x n x sc/n), i.e. flattens last 2 layers
        self.reshape_2 = tf.keras.layers.Reshape((n_hash, -1))
        # hash functions
        self.hashers = [tf.keras.layers.Dense(hashdim, activation='tanh') for _ in range(n_hash)]
    def call(self, x):
        x = self.channel_remap(x)
        x = self.reshape_1(x)
        x = self.permuter(x)
        x = self.reshape_2(x)
        # get all hashes then reassemble
        hashes = []
        for i, hasher in enumerate(self.hashers):
            hashes.append(hasher(x[:,i,:])[:,None,:]) # get slice of x then add back dimension
        return tf.concat(hashes, 1)

# MODELS

class ChunkHash:
    def __init__(self, tokens, chunksz=1024, d_model=256, hashsz=64, cr_layers=4):
        self.tokens = tokens
        self.d_model = d_model
        self.cr_layers = cr_layers
        self.c = chunksz
        self.h = hashsz
        d_emb = d_model

        # use half precision
        #tf.keras.mixed_precision.set_global_policy('float16')

        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d_emb) # b x l x d
        # feature extraction
        self.extractor = ResBlock1D(dim=d_emb, kernelsz=9)
        # conv/pool
        #self.pool = tf.keras.layers.MaxPooling1D(chunksz, chunksz//2) # b x 2l//c x d
        self.pool = ConvReduceBlock(chunksz, cr_layers, d_emb)
        # hash generation
        self.hash_layer = tf.keras.layers.Dense(hashsz, activation='tanh')
        self.distributed_hash = tf.keras.layers.TimeDistributed(self.hash_layer)

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(None,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(None,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            emb = self.kmer_emb(ins)

            emb_feats = self.extractor(emb)
            chunk_feats = self.pool(emb_feats)
            hashes_raw = self.distributed_hash(chunk_feats)
            # add dimension
            hashes.append(hashes_raw)
        
        hash_loss, hash_precision, hash_recall, _ = hash_metric_factory(self.h)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall]
                           )


# discretized chunk hash: break sequence into separate chunks before feature extraction
class DiscretizedChunkHash:
    def __init__(self, tokens, chunksz=1024, d_model=256, hashsz=64):
        self.tokens = tokens
        self.d_model = d_model
        self.c = chunksz
        self.h = hashsz
        d_emb = d_model

        # TODO: get nearest chunksz factorizable into small primes 

        # use half precision
        #tf.keras.mixed_precision.set_global_policy('float16')

        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d_emb) # b x l x d
        # feature extraction
        self.extractor = ResBlock1D(dim=d_emb, kernelsz=9, layers=4)
        # conv/pool
        #self.pool = tf.keras.layers.MaxPooling1D(chunksz, chunksz//2) # b x 2l//c x d
        self.pool = ChunkReduceBlock(chunksz, d_emb)
        # hash generation
        #self.hash_layer = tf.keras.layers.Dense(hashsz, activation='tanh')
        self.hash_layer = MLPHasher(hashsz)
        self.distributed_hash = tf.keras.layers.TimeDistributed(self.hash_layer)

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(None,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(None,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            # pad
            length = tf.shape(ins)[-1]
            chunk_count = length // self.c
            #to_add = length % self.c
            # this has to be evil because length is a TF symbolic value
            #ins = tf.pad(ins, tf.concat([tf.constant([[0,0]]),tf.stack([[tf.constant(0)],[to_add]], axis=-1)], axis=0))
            #length = length + to_add

            # get mask from input tokens
            mask = tf.not_equal(ins, 0)
            # convert to "is this chunk empty?" + add dim
            chunk_mask = mask[:,::self.c,None]

            # get embeddings
            emb = self.kmer_emb(ins)

            # break into chunks + flatten first 2 dims
            emb = tf.reshape(emb, tf.stack([-1, self.c, self.d_model]))

            # TODO: drop mostly (>50%) empty chunks

            # extract features
            emb_feats = self.extractor(emb)

            # sort by norm
            emb_feats_norm = tf.norm(emb_feats, axis=-1)
            sorted_idx = tf.argsort(emb_feats_norm)
            emb_feats = tf.gather(emb_feats, sorted_idx, axis=-2, batch_dims=1)

            # reduce each chunk to 1 feature set
            chunk_feats = self.pool(emb_feats)

            # fix dims
            #chunk_feats = tf.reduce_mean(chunk_feats, axis=-2)
            chunk_feats = tf.squeeze(chunk_feats)
            chunk_feats = tf.reshape(chunk_feats, tf.stack([-1, chunk_count, self.d_model]))

            # hash
            hashes_raw = self.distributed_hash(chunk_feats)
            # zero out hashes from empty chunks
            hashes_raw *= tf.cast(chunk_mask, float)
            # add dimension
            hashes.append(hashes_raw)

        hash_loss, hash_precision, hash_recall, unique = hash_metric_factory(self.h, alpha=0.2)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall, unique]
                           )
        
    # single chunk generator model
    def compile_single(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            # get embeddings (b x c x d)
            emb = self.kmer_emb(ins)

            # extract features
            emb_feats = self.extractor(emb)

            # sort by norm
            #emb_feats_norm = tf.norm(emb_feats, axis=-1)
            #sorted_idx = tf.argsort(emb_feats_norm)
            #emb_feats = tf.gather(emb_feats, sorted_idx, axis=-2, batch_dims=1)

            # reduce each chunk to 1 feature set (b x 1 x d)
            chunk_feats = self.pool(emb_feats)
            chunk_feats = tf.squeeze(chunk_feats, axis=-2) # (b x d)

            # hash (b x h); no need for distributed version
            hashes_raw = self.hash_layer(chunk_feats)
            hashes.append(hashes_raw)

        hash_loss, hash_precision, hash_recall, unique = hash_metric_factory_single(self.h, w_bal=0.1)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall, unique]
                           )
        
    def make_inference_model(self, min_pop=0.8):
        inseqs = tf.keras.layers.Input(shape=(None,), dtype='int32')

        length = tf.shape(inseqs)[-1]
        chunk_count = length // self.c

        # get mask from input tokens
        mask = tf.not_equal(inseqs, 0)
        # convert to "is this chunk sufficiently populated?" + add dim
        chunk_mask = mask[:,int(self.c * min_pop)::self.c,None]

        # get embeddings
        emb = self.kmer_emb(inseqs)

        # break into chunks + flatten first 2 dims
        emb = tf.reshape(emb, tf.stack([-1, self.c, self.d_model]))

        # extract features
        emb_feats = self.extractor(emb)
        chunk_feats = self.pool(emb_feats)

        # fix dims
        chunk_feats = tf.squeeze(chunk_feats)
        chunk_feats = tf.reshape(chunk_feats, tf.stack([-1, chunk_count, self.d_model]))

        # hash
        hashes_raw = self.distributed_hash(chunk_feats)
        hashes = tf.sign(hashes_raw)
        hashes *= tf.cast(chunk_mask, float)

        self.hasher = tf.keras.Model(inseqs, hashes)

# multiple hashes per chunk, require only 1 hit
# horrible idea: name this ChunkBucket
class ChunkMultiHash:
    def __init__(self, tokens, chunksz=1024, d_model=256, hashsz=64, n_hash=32):
        self.tokens = tokens
        self.d_model = d_model
        self.c = chunksz
        self.h = hashsz
        self.n = n_hash
        d = d_model

        # kmer embedding
        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d) # b x l x d

        # encoder convnet
        self.conv_pool = tf.keras.layers.MaxPool1D(2)
        self.conv_layers = [tf.keras.layers.Conv1D(d, 32, activation='relu', padding='same'), self.conv_pool,
                            InceptionLayer(d), self.conv_pool,
                            InceptionLayer(d*2), self.conv_pool, 
                            InceptionLayer(d*4), self.conv_pool,
                            ResBlock1D(dim=d*8, kernelsz=4, layernorm=True),
                            ]

        # hasher
        self.flatten = tf.keras.layers.Flatten()
        self.hash_layer = tf.keras.layers.Dense(hashsz*n_hash, activation='tanh')

    def compile(self, optimizer='adam', mode='train'):
        inseqs1 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            # get embeddings (b x c x d)
            x = self.kmer_emb(ins)

            # encode with convnet; output (b x s x d), where s is some # of downsampled steps
            for layer in self.conv_layers:
                x = layer(x)

            # normalize -> hash
            x = tf.math.l2_normalize(x, axis=-1)
            x = self.flatten(x)
            x = self.hash_layer(x)
            hashes_raw = tf.reshape(x, [-1, self.n, self.h])
            
            hashes.append(hashes_raw)

        hash_loss, hash_precision, hash_recall, unique = bucket_metric_factory(self.n, self.h, w_bal=0.0)

        metrics = []
        if mode == 'train':
            metrics = [hash_precision, hash_recall, unique]
        elif mode == 'eval':
            TNR = bucket_TNR_factory(self.h)
            metrics = [hash_recall, TNR, unique]


        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           metrics
                           )
    
    # create single sequence hashing model, store as self.hasher
    def make_inference_model(self):
        inseqs = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        x = self.kmer_emb(inseqs)
        for layer in self.conv_layers:
            x = layer(x)

        x = tf.math.l2_normalize(x, axis=-1)
        x = self.flatten(x)
        x = self.hash_layer(x)
        hashes = tf.reshape(x, [-1, self.n, self.h])

        self.hasher = tf.keras.Model(inseqs, hashes)


if __name__ == '__main__':
    print('done')
