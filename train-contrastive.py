# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import tfv2transformer.input as dd
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

# set global max length, batch size, and k
max_len = 2048
batch_size = 128
k = 8

itokens, otokens = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
gen = dd.gen_simple_contrastive_data(max_len=max_len, batch_size=batch_size, tokens=itokens, k=k)
#gen = dd.KmerDataGenerator('/fs/nexus-scratch/rhaworth/hmp-mini/', itokens, otokens, batch_size=4, max_len=max_len)

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num()) # we don't use this here, go back and fix later

from tfv2transformer.transformer_sparse import LRSchedulerPerStep
from tfv2transformer.contrastive_encoder import ContrastiveEncoder

d_model = 512
block_size = 64
enc = ContrastiveEncoder(itokens, len_limit=1024, d_model=d_model, d_inner_hid=512, \
                   n_head=8, layers=2, length=max_len, block_size=block_size, dropout=0.1)

mfile = '/fs/nexus-scratch/rhaworth/models/contrastive.model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)
model_saver = ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

enc.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9), batch_size=batch_size)
try: enc.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
    print('not implemented')
elif 'test' in sys.argv:
    print('not implemented')
else:
    enc.model.summary()
    if not os.path.isdir('models'): os.mkdir('models')
    enc.model.fit(gen, steps_per_epoch=200, epochs=10, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler, model_saver])
    # TODO: check some sequences
    print('done training')