# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import tfv2transformer.input as dd
from tfv2transformer.gensynth import gen_simple_block_data_binary
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

# set global max and min length, batch size, and k
max_len = 4096
min_len = max_len//4
batch_size = 32
k = 4

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
gen = gen_simple_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)

print('kmer dict size:', itokens.num())

from tfv2transformer.transformer_sparse import LRSchedulerPerStep
from tfv2transformer.skew_attn import SimpleSkewBinary

d_model = 128
ssb = SimpleSkewBinary(itokens, d_model=d_model, length=max_len)

mfile = '/fs/nexus-scratch/rhaworth/models/skew.model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)
model_saver = ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

ssb.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try: ssb.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
    print('not implemented')
elif 'test' in sys.argv:
    print('not implemented')
else:
    ssb.model.summary()
    if not os.path.isdir('models'): os.mkdir('models')
    ssb.model.fit(gen, steps_per_epoch=100, epochs=15, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler, model_saver]
                )
    print('done training')
    # check accuracy
    ssb.model.evaluate(gen, steps=5)
