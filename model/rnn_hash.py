# rnn encoder
import tensorflow as tf
from .hash_metrics import bucket_metric_factory

class GRUHash:
    def __init__(self, tokens, chunksz=1024, d_model=256, hashsz=64, n_hash=32):
        self.tokens = tokens
        self.d_model = d_model
        self.c = chunksz
        self.h = hashsz
        self.n = n_hash
        d = d_model

        # kmer embedding
        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d)

        # gru encoder
        self.encoder = tf.keras.layers.GRU(d)
        
        # hasher
        self.flatten = tf.keras.layers.Flatten()
        self.hash_layer = tf.keras.layers.Dense(hashsz*n_hash, activation='tanh')

    def compile(self, optimizer='adam'):
        inseqs1 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        inseqs2 = tf.keras.layers.Input(shape=(self.c,), dtype='int32')

        inseqs = [inseqs1, inseqs2]

        # encode and hash
        hashes = []
        for x in inseqs:
            # encode
            x = self.kmer_emb(x)
            x = self.encoder(x)
            #x = tf.math.l2_normalize(x, axis=-1) # for some reason this continues to be critical to the model not exploding
            # hash
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