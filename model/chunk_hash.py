# learning to hash chunks of sequences such that similar sequences tend to produce collisions
# some sections modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf

# METRICS

# add loss/metric hyperparameters here
def hash_metric_factory(h=64, alpha=0.2, w_neg=1.0):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x n x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        #ht1 = tf.sign(ht1)
        #ht2 = tf.sign(ht2)

        # compute (b x n x n) dot product similarity tensor
        sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # assume sim = 0.0 means at least one hash is invalid
        #sims = tf.where(sims == 0.0, -1.0, sims)

        # find max for each (n x n) dot product matrix
        max_sim = tf.reduce_max(sims, axis=[-2,-1])
        #min_sim = tf.reduce_min(sims, axis=[-2,-1])
        mean_sim = tf.reduce_mean(sims, axis=[-2,-1])
        
        # HashNet loss, slightly modified
        #loss = tf.math.log(1+tf.math.exp(alpha*max_sim)) - tf.cast(y_true, float) * max_sim * alpha
        loss_pos = tf.math.log(1+tf.math.exp(alpha*max_sim)) - max_sim * alpha
        loss_neg = tf.math.log(1+tf.math.exp(alpha*(mean_sim + max_sim)))

        loss = tf.where(y_true == 1, loss_pos, loss_neg * w_neg) # type: ignore

        return tf.reduce_sum(loss)

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

    # set names
    hash_precision.__name__ = 'precision'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall

# COMPONENTS

# reduce entire chunk to 1 feature map
class ChunkReduceBlock(tf.keras.layers.Layer):
    def __init__(self, chunksz=1024, dim=128):
        super(ChunkReduceBlock, self).__init__()
        # assemble convs to perform reduction
        # assume chunksz can be mostly factorized by these, i.e. the largest prime factor is small
        factors = [2,3,4,5]
        currentsz = chunksz
        self.redconvs = []
        for factor in factors[::-1]:
            while currentsz % factor == 0:
                # maybe these don't need to have a bigger filter than the stride, test later
                currentsz = currentsz // factor
                self.redconvs.append(tf.keras.layers.Conv1D(dim, factor, factor,
                                                            padding='valid', activation='tanh'))
        # extra conv if needed
        if currentsz > 1:
            self.redconvs.append(tf.keras.layers.Conv1D(dim, currentsz, currentsz,
                                                        padding='valid', activation='tanh'))
        # resblocks
        self.resblocks = [ResBlock(dim=dim, layers=4) for _ in range(len(self.redconvs))]
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
        self.resblocks = [ResBlock(dim=dim, layers=4) for _ in range(len(self.redconvs))]
    def call(self, x):
        for res, red in zip(self.resblocks, self.redconvs):
            x = res(x)
            x = red(x)
        return x

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, layers=2, dim=128, kernelsz=3, activation='relu', layernorm=False):
        super(ResBlock, self).__init__()
        # create convolutional layers
        self.layernorm = None
        self.convs = [tf.keras.layers.Conv1D(dim, kernelsz, activation=activation, padding='same') for _ in range(layers)]
        if layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x_init = x
        for conv in self.convs:
            x = conv(x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        return x + x_init
    

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
        self.extractor = ResBlock(dim=d_emb, kernelsz=9)
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
        
        hash_loss, hash_precision, hash_recall = hash_metric_factory(self.h)

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
        self.extractor = ResBlock(dim=d_emb, kernelsz=9)
        # conv/pool
        #self.pool = tf.keras.layers.MaxPooling1D(chunksz, chunksz//2) # b x 2l//c x d
        self.pool = ChunkReduceBlock(chunksz, d_emb)
        # hash generation
        self.hash_layer = tf.keras.layers.Dense(hashsz, activation='tanh')
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

        hash_loss, hash_precision, hash_recall = hash_metric_factory(self.h, alpha=0.2)

        # create model
        self.model = tf.keras.Model(inseqs, tf.stack(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall]
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

if __name__ == '__main__':
    print('done')
