# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import model.input as dd
import model.gensynth as gs
import numpy as np
import tensorflow as tf

# set global max and min length, batch size, and k
max_len = 15000
min_len = 1000
batch_size = 32 #32*16
k = 4

d_model = 64
chunksz = 2000
hashsz = 64

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = gs.gen_adversarial_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
gen = gs.gen_simple_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)

print('kmer dict size:', itokens.num())

from model.chunk_hash import DiscretizedChunkHash, ChunkHash

ssb = DiscretizedChunkHash(itokens, chunksz=chunksz, d_model=d_model, hashsz=hashsz)

def lr_schedule(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * np.exp(-1)

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
    ssb.model.fit(gen, steps_per_epoch=100, epochs=10, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler, model_saver])
    print('done training')
    # check accuracy
    print('train gen eval')
    ssb.model.evaluate(gen, steps=100)

    """
    print('simple gen eval')
    gensimple = gs.gen_simple_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
    ssb.model.evaluate(gensimple, steps=100)"""
