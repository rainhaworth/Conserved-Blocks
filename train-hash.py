# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import model.input as dd
import model.gensynth as gs
import numpy as np
import tensorflow as tf

# set global max and min length, batch size, and k
max_len = 500000
min_len = 1000
batch_size = 128 #1024
k = 4

d_model = 256
chunksz = 2000
hashsz = 64
n_hash = 4 # for bucketing approach

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = gs.gen_adversarial_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
#gen = gs.gen_simple_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
gen = gs.gen_adversarial_chunks_binary(chunk_size=chunksz, min_shared=min_len, batch_size=batch_size,
                                       tokens=itokens, k=k, boundary_pad=20, min_pop=1.0)

print('kmer dict size:', itokens.num())

from model.chunk_hash import DiscretizedChunkHash, ChunkHash, ChunkMultiHash

ssb = ChunkMultiHash(itokens, chunksz=chunksz, d_model=d_model, hashsz=hashsz, n_hash=n_hash)

def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * np.exp(-0.5)

mfile = '/fs/nexus-scratch/rhaworth/models/chunkhash.model.h5'

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
model_saver = tf.keras.callbacks.ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

ssb.compile()
try: ssb.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
    print('not implemented')
elif 'test' in sys.argv:
    print('not implemented')
else:
    ssb.model.summary()
    ssb.model.fit(gen, steps_per_epoch=100, epochs=15, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler])
    print('done training')

    # check accuracy on original setting
    print('train gen eval')
    ssb.model.evaluate(gen, steps=100)

    # check on simpler setting
    print('simple gen eval')
    gensimple = gs.gen_simple_block_data_binary(max_len=chunksz, min_len=chunksz, block_min=min_len, batch_size=batch_size, tokens=itokens, k=k)
    ssb.model.evaluate(gensimple, steps=100)
