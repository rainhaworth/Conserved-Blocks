# learning to hash chunks of sequences such that similar sequences tend to produce collisions
# some sections modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf
import numpy as np

# METRICS

# add loss/metric hyperparameters here
def hash_metric_factory(h=64, alpha=0.2, w_neg=1.0, w_bal=0.5):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x n x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute (b x n x n) dot product similarity tensor
        sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # find max for each (n x n) dot product matrix
        max_sim = tf.reduce_max(sims, axis=[-2,-1])
        #min_sim = tf.reduce_min(sims, axis=[-2,-1])
        mean_sim = tf.reduce_mean(sims, axis=[-2,-1])
        
        # HashNet loss, slightly modified
        loss_pos = tf.math.log(1+tf.math.exp(alpha*max_sim)) - max_sim * alpha
        loss_neg = tf.math.log(1+tf.math.exp(alpha*(mean_sim + max_sim)))
        loss_hn = tf.where(y_true == 1, loss_pos, loss_neg * w_neg) # type: ignore

        loss = tf.reduce_sum(loss_hn)

        # hash balance loss: try to push each hash in a sequence to have different values
        # (ideally want all hashes in each half-batch to have different values but that's expensive)

        for ht in (ht1, ht2):
            #(b x n x h) -> (b x n x n)
            loss_bal = tf.matmul(ht, ht, transpose_b=True)
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1]), padding_value=-1) # subtract I
            loss += tf.reduce_mean(loss_bal) * w_bal

        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])

        # get similarity values from 0 to 1
        hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # get max similarity
        max_sim = tf.reduce_max(hash_sims, axis=[-2,-1])

        # predicted positives
        exact_match = tf.where(max_sim == 1.0,
                                max_sim,
                                0 * max_sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        y_true = tf.squeeze(y_true)

        # get similarity values from 0 to 1
        hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # get max sim
        max_sim = tf.reduce_max(hash_sims, axis=[-2,-1])

        exact_match = tf.where(max_sim == 1.0,
                                max_sim,
                                0 * max_sim)
        
        # get true positives
        tp = exact_match * y_true

        # divide positive sample similarities by # positive samples
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)

        # average hash similarity
        #return tf.reduce_mean(hash_sims)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.sign(y_pred)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

# metrics for single chunk regime
def hash_metric_factory_single(h=64, alpha=0.2, w_neg=1.0, w_bal=0.5):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        
        """
        # HashNet loss, slightly modified
        loss_pos = tf.math.log(1+tf.math.exp(alpha*sim)) - sim * alpha
        loss_neg = tf.math.log(1+tf.math.exp(alpha*sim))
        loss_hn = tf.where(y_true == 1, loss_pos, loss_neg * w_neg) # type: ignore"""

        # simplified loss
        y_true = tf.cast(y_true, float) * 2.0 - 1.0 # type: ignore
        #loss_hn = tf.abs(sim - y_true)

        # new loss: HN w/ pos and neg equally weighted
        loss_hn = tf.math.log(1+tf.math.exp(-y_true * sim * 2))

        loss = tf.reduce_mean(loss_hn)

        # intra-batch loss
        for ht in (ht1, ht2):
            #(b x h) -> (b x b)
            # hash balance loss: try to push each hash in each batch to have different values
            loss_bal = tf.matmul(ht, ht, transpose_b=True) / h
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1])) # subtract I; removed padding_value = -1
            loss += tf.reduce_mean(loss_bal) * w_bal

            # bit uncorrelation loss
            loss_unc = ht * tf.ones_like(ht)
            loss_unc = tf.reduce_mean(loss_unc) ** 2
            loss += loss_unc * w_bal


        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # get predicted positives
        exact_match = tf.where(sim == 1.0,
                                sim,
                                0 * sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        y_true = tf.squeeze(y_true)

        # get similarity values from -1 to 1
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # get predicted positives
        exact_match = tf.where(sim == 1.0,
                                sim,
                                0 * sim)
        
        # get true positives
        tp = exact_match * y_true

        # divide positive sample similarities by # positive samples
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.sign(y_pred)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

# metrics for bucketing
def bucket_metric_factory(n=32, h=64, alpha=0.2, w_neg=1.0, w_bal=0.5):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x n x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute similarity for each hash pair; this should still work, output (b x n)
        #sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # TODO: roll + tile so we can get a distance matrix between all hashes

        # compute euclidean (l2 norm) distance, get min
        dist = tf.norm(ht1 - ht2, axis=-1) # (b x n)
        min_dist = tf.reduce_min(dist, axis=-1) # (b)

        # softplus loss
        z = min_dist - 1.0 # intentionally using 0.5 instead of 1.0 to shift to higher recall
        y_true = tf.cast(y_true, float) * 2.0 - 1.0 # type: ignore
        loss_hn = tf.math.log(1+tf.math.exp(y_true * z))
        #loss_hn = tf.maximum(tf.zeros_like(z), 1+y_true*z)

        # hinge loss
        #loss_hn = tf.maximum(tf.zeros_like(z), 1 + z * y_true)

        loss = loss_hn #tf.reduce_mean(loss_hn)

        # intra-batch loss
        """
        for ht in (ht1, ht2):
            ht = tf.reshape(ht, [-1, h])
            #(b x h) -> (b x b)
            # hash balance loss: try to push each hash in each batch to have different values
            loss_bal = tf.matmul(ht, ht, transpose_b=True) / h
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1])) # subtract I; removed padding_value = -1
            loss += tf.reduce_mean(loss_bal) * w_bal

            # bit uncorrelation loss
            loss_unc = ht * tf.ones_like(ht)
            loss_unc = tf.reduce_mean(loss_unc) ** 2
            loss += loss_unc * w_bal"""

        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        #ht1 = tf.where(y_pred[0] > 0.5, 1.0, -1.0)
        #ht2 = tf.where(y_pred[1] > 0.5, 1.0, -1.0)

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        max_sim = tf.reduce_max(sim, axis=-1)
        

        # get predicted positives
        exact_match = tf.where(max_sim == 1.0,
                               max_sim,
                               0 * max_sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        #ht1 = tf.where(y_pred[0] > 0.5, 1.0, -1.0)
        #ht2 = tf.where(y_pred[1] > 0.5, 1.0, -1.0)
        y_true = tf.squeeze(y_true)

        # get similarity values from -1 to 1
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        max_sim = tf.reduce_max(sim, axis=-1)

        # get predicted positives
        exact_match = tf.where(max_sim == 1.0,
                               max_sim,
                               0 * max_sim)
        
        # get true positives
        tp = exact_match * y_true

        # divide positive sample similarities by # positive samples
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.where(y_pred > 0.5, 1.0, -1.0)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

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
    def __init__(self, dim=128, minlen=1, maxlen=9, pool=2, layernorm=False, reduce=True):
        super(InceptionLayer, self).__init__()

        self.dim = dim
        if reduce:
            # require divisibility by # of filters
            assert (dim*pool) % (maxlen-minlen) == 0
            self.dim = (dim*pool) // (maxlen-minlen)

        self.filters = [tf.keras.layers.Conv1D(self.dim, f, activation='relu', padding='same')
                        for f in range(minlen, maxlen)]
        self.pool = tf.keras.layers.MaxPool1D(pool)

        self.reducers = [None for _ in range(len(self.filters))]
        if reduce:
            # first layer does not need a reducer
            self.reducers[1:] = [tf.keras.layers.Conv1D(self.dim, 1, activation='relu') for _ in range(len(self.filters)-1)]
    
    def call(self, x):
        # apply filters
        out = []
        for reducer, filter in zip(self.reducers, self.filters):
            o = x
            if reducer != None:
                o = reducer(o)
            o = filter(o)
            o = self.pool(o)
            out.append(o)
        # concat feature maps
        return tf.concat(out, axis=-1)

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

        #assert (d * 16) % n_hash == 0

        # kmer embedding
        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d) # b x l x d

        # feature extraction
        # init conv
        self.conv_1 = tf.keras.layers.Conv1D(d, 8, activation='relu', padding='same')
        # dim reduction
        self.conv_pool = tf.keras.layers.MaxPool1D(2)
        # sequential convnet
        self.conv_layers = [self.conv_1, self.conv_pool, 
                            ResBlock1D(dim=d, kernelsz=8, layernorm=True), self.conv_pool,
                            #ResBlock1D(dim=d, kernelsz=8, layernorm=True), self.conv_pool,
                            #ResBlock1D(dim=d, kernelsz=4, layernorm=True), self.conv_pool, 
                            #ResBlock1D(dim=d, kernelsz=4, layernorm=True), self.conv_pool, 
                            ResBlock1D(dim=d, kernelsz=8, layernorm=True)]

        # edge extractors; outputs concatenated
        #self.edge_layers = [tf.keras.layers.Conv1D(d, f, activation='relu') for f in range(2, 10)]
        self.inception = InceptionLayer(d)

        # reduce to ~original size
        self.edge_pool = tf.keras.layers.MaxPool1D(2)
        #self.edge_norm = tf.keras.layers.LayerNormalization()

        # fix dims for TimeDistributed
        self.permuter = tf.keras.layers.Permute((2,1,3))

        # hasher
        self.flatten = tf.keras.layers.Flatten()
        self.hash_layer = SplitHasher(hashsz, n_hash, d)
        #self.hash_layer = tf.keras.layers.Dense(hashsz*n_hash, activation='tanh') # for non-distributed do hashsz*n_hash
        #self.hash_layer = MLPHasher(hashsz*n_hash)
        self.distributed_hash = tf.keras.layers.TimeDistributed(self.hash_layer)

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            # make embedding mask
            #mask = tf.not_equal(ins, 0)[...,None]
            #mask = tf.cast(mask, float)

            # get embeddings (b x c x d)
            x = self.kmer_emb(ins)
            # apply mask
            #x = x * mask

            # convnet; output (b x s x d), where s is some # of downsampled steps
            for layer in self.conv_layers:
                x = layer(x)

            # apply inception layer
            x = self.inception(x)
            x = tf.math.l2_normalize(x, axis=-1)
            hashes_raw = self.hash_layer(x)
            #x = self.flatten(x)
            #x = self.hash_layer(x)
            #hashes_raw = tf.reshape(x, [-1, self.n, self.h])
            
            """
            # reshape to (b x s x n x (d*16//n))
            s = tf.shape(x)[1]
            x = tf.reshape(x, [-1, s, self.n,
                               self.d_model//self.n])
            # rearrange dims
            x = self.permuter(x)
            # reshape to (b x n x (s*d*16//n)); essentially flatten last 2 layers
            x = tf.reshape(x, [-1, self.n, s*self.d_model//self.n])

            # get hashes (b x n x h)
            hashes_raw = self.distributed_hash(x)

            ###############

            # edges
            edges = []
            for el in self.edge_layers:
                e = el(x)
                e = self.edge_pool(e)
                edges.append(e)
            # concat
            x = tf.concat(edges, axis=-2)

            # normalize; avoids nan loss
            x = tf.math.l2_normalize(x, axis=-1)

            # big hash
            x = self.flatten(x)
            x = self.hash_layer(x)
            hashes_raw = tf.reshape(x, [-1, self.n, self.h])"""
            hashes.append(hashes_raw)

        hash_loss, hash_precision, hash_recall, unique = bucket_metric_factory(self.n, self.h, w_bal=0.0)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall, unique]
                           )


if __name__ == '__main__':
    print('done')
