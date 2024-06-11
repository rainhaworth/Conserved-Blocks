# skewed attention mechanism for identifying sequences with shared regions
# with downsampling for memory footprint reduction
# modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf

class ConvReduceBlock(tf.keras.layers.Layer):
    def __init__(self, downsample=1024, layers=4, dim=128):
        super(ConvReduceBlock, self).__init__()
        # reduce 'downsample' kmers to 1 across N layers
        self.ds_per_layer = int(downsample ** (1 / layers))
        self.convs = [tf.keras.layers.Conv1D(dim, self.ds_per_layer * 2 + 1, self.ds_per_layer, activation='tanh') for _ in range(layers)]
        # extra conv if needed
        ds_current = self.ds_per_layer ** layers
        if ds_current != downsample:
            ds_last = downsample // ds_current
            self.convs.append(tf.keras.layers.Conv1D(dim, ds_last * 2 + 1, ds_last, activation='tanh'))
    def call(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, layers=2, dim=128, kernelsz=3, activation='relu'):
        super(ResBlock, self).__init__()
        # create convolutional layers
        self.convs = [tf.keras.layers.Conv1D(dim, kernelsz, activation=activation, padding='same') for _ in range(layers)]
    def call(self, x):
        x_init = x
        for conv in self.convs:
            x = conv(x)
        return x + x_init

class ChunkHash:
    def __init__(self, tokens, chunksz=1024, d_model=256, hashsz=64):
        self.tokens = tokens
        self.d_model = d_model
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
        self.pool = ConvReduceBlock(chunksz//2, 4, d_emb)
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
            hashes.append(hashes_raw[None,...])

        # loss given pairwise labels
        def hash_loss(y_true, y_pred):
            # y_true: (b) labels
            # y_pred: list of 2 (b x n x h) hash tensors
            y_true = tf.squeeze(y_true)

            # get hash tensors
            ht1 = y_pred[0]
            ht2 = y_pred[1]

            # compute (b x n x n) dot product similarity tensor
            sims = tf.matmul(ht1, ht2, transpose_b=True) / self.h

            # find max for each (n x n) dot product matrix
            max_sim = tf.reduce_max(sims, axis=[-2,-1])
            #mean_sim = tf.reduce_mean(sims, axis=[-2,-1])
            
            # HashNet loss w/ w_ij = 1, alpha = 1
            alpha = 0.2
            #loss = tf.math.log(1+tf.math.exp(alpha*max_sim)) - tf.cast(y_true, float) * max_sim * alpha
            loss_pos = tf.math.log(1+tf.math.exp(alpha*max_sim)) - max_sim * alpha
            loss_neg = tf.math.log(1+tf.math.exp(alpha*max_sim)) # tried using mean_sim but it performs worse

            loss = tf.where(y_true == 1, loss_pos, loss_neg)

            return tf.reduce_sum(loss)

        # precision: fraction of samples with at least one exact match that are positives
        def hash_precision(y_true, y_pred):
            # get hash tensors; convert to binary
            ht1 = tf.sign(y_pred[0])
            ht2 = tf.sign(y_pred[1])

            # get similarity values from 0 to 1
            hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / self.h

            # get max similarity
            max_sim = tf.reduce_max(hash_sims, axis=[-2,-1])

            # predicted positives
            exact_match = tf.where(max_sim == 1.0,
                                   max_sim,
                                   0 * max_sim)

            # true positives
            tp = exact_match * tf.squeeze(y_true)

            # return mean
            return tf.reduce_sum(tp) / tf.reduce_sum(exact_match)
        
        # recall: fraction of positive samples with at least one exact match
        def hash_recall(y_true, y_pred):
            # get hash tensors; convert to binary
            ht1 = tf.sign(y_pred[0])
            ht2 = tf.sign(y_pred[1])
            y_true = tf.squeeze(y_true)

            # get similarity values from 0 to 1
            hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / self.h

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

        # create model
        self.model = tf.keras.Model(inseqs, tf.concat(hashes, 0))
        self.model.compile(optimizer,
                           hash_loss, 
                           [hash_precision, hash_recall]
                           )

if __name__ == '__main__':
    print('done')
