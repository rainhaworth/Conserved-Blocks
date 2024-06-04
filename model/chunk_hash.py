# skewed attention mechanism for identifying sequences with shared regions
# with downsampling for memory footprint reduction
# modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf

class MLP(tf.keras.layers.Layer):
    def __init__(self, layers=4, dim=64, predlast=False):
        super(MLP, self).__init__()
        self.NNs = [tf.keras.layers.Dense(dim, activation='relu') for _ in range(layers)]
        if predlast:
            self.NNs.append(tf.keras.layers.Dense(dim))
    def call(self, x):
        for layer in self.NNs:
            x = layer(x)
        return x

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
        self.pool = tf.keras.layers.AveragePooling1D(chunksz, chunksz//2) # b x 2l//c x d
        self.hash_layer = MLP(dim=hashsz, predlast=True)
        self.distributed_hash = tf.keras.layers.TimeDistributed(self.hash_layer)
        self.tanh = tf.keras.layers.Activation('tanh', name='h')

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(None,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(None,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        hashes = []
        
        for ins in inseqs:
            emb = self.kmer_emb(ins)
            chunk_feats = self.pool(emb)
            feats_sorted = tf.sort(chunk_feats, axis=-1)
            hashes_raw = self.distributed_hash(feats_sorted)
            hashes.append(self.tanh(hashes_raw))

        # loss given pairwise labels
        def hash_loss(y_true, y_pred):
            # y_true: (b) labels
            # y_pred: list of 2 (b x n x h) hash tensors

            # get hash tensors
            ht1 = y_pred[0]
            ht2 = y_pred[1]

            # compute (b x n x n) dot product similarity tensor
            sims = tf.matmul(ht1, ht2, transpose_b=True)

            # positive samples: find maximum similarity, convert to hamming distance for minimization
                # dist_H = 1/2 * (hashsz - sim)
            # negative samples: minimize norm of similarity matrix
            loss = tf.where(y_true == 1,
                            0.5 * (self.h - tf.reduce_max(sims, axis=[-2,-1])),
                            tf.reduce_max(sims, axis=[-2,-1]))

            return loss

        # hash accuracy metric
        def hash_acc(y_true, y_pred):
            # get hash tensors; convert to binary
            ht1 = tf.sign(y_pred[0])
            ht2 = tf.sign(y_pred[1])

            # get similarity values from 0 to 1
            hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / self.h

            # positive samples: how close is max to 1?
            # negative samples: how close is max to 0?
            max_hash_sims = tf.reduce_max(hash_sims, axis=[-2,-1])
            dists = 1 - tf.abs(y_true - max_hash_sims)

            # return mean
            return tf.reduce_mean(dists)
        
        # hash recall metric
        def hash_recall(y_true, y_pred):
            # get hash tensors; convert to binary
            ht1 = tf.sign(y_pred[0])
            ht2 = tf.sign(y_pred[1])

            # get similarity values from 0 to 1
            hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / self.h

            max_hash_sims = tf.reduce_max(hash_sims, axis=[-2,-1])
            mhs_positive = max_hash_sims * y_true # mask

            # divide positive sample similarities by # positive samples
            return tf.reduce_sum(mhs_positive) / tf.reduce_sum(y_true)
        
        # set names
        hash_acc.__name__ = 'ha'
        hash_recall.__name__ = 'hr'

        # create model
        self.model = tf.keras.Model(inseqs, tf.concat(hashes, 1))
        self.model.compile(optimizer,
                     hash_loss, 
                     [hash_acc, hash_recall]
                     )

if __name__ == '__main__':
    print('done')
