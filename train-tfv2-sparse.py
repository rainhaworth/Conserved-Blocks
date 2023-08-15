# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import pipelines.tfv2trans_input as dd
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

# set global max length
max_len = 4096

itokens, otokens = dd.LoadKmerDict('./utils/8mers.txt')
gen = dd.KmerDataGenerator('./data-tmp/', itokens, otokens, batch_size=4, max_len=max_len)
#/fs/nexus-scratch/rhaworth/hmp-mini/

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from tfv2transformer.transformer_sparse import Transformer, LRSchedulerPerStep

d_model = 512
block_size = 64
s2s = Transformer(itokens, otokens, len_limit=1024, d_model=d_model, d_inner_hid=512, \
                   n_head=8, layers=2, length=max_len, block_size=block_size, dropout=0.1)

mfile = 'models/tmp.model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000) 
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
    # not implemented
    for x, y in s2s.beam_search('A big dog eats food .'.split(), delimiter=' '):
        print(x, y)
    print(s2s.decode_sequence_readout('A big dog eats food .'.split(), delimiter=' '))
    print(s2s.decode_sequence_fast('A big dog eats food .'.split(), delimiter=' '))
    while True:
        quest = input('> ')
        print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
        rets = s2s.beam_search(quest.split(), delimiter=' ')
        for x, y in rets: print(x, y)
elif 'test' in sys.argv:
    # not implemented
    import tfv2transformer.ljqpy as ljqpy
    valids = ljqpy.LoadCSV('data/en2de.s2s.valid.txt')
    en = [x[0].split() for x in valids[:100]]
    rets = s2s.decode_sequence_readout(en, delimiter=' ')
    for x in rets[:5]: print(x)

    rets = s2s.beam_search(en, delimiter=' ', verbose=1)
    for i, x in enumerate(rets[:5]):
        print('-'*20)
        print(valids[i][1])
        for y in x: print(y)

    rets = s2s.decode_sequence_fast(en, delimiter=' ', verbose=1)
    for x in rets[:5]: print(x)
else:
    s2s.model.summary()
    if not os.path.isdir('models'): os.mkdir('models')
    s2s.model.fit(gen, steps_per_epoch=200, epochs=10, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler, model_saver])
    # check output
    preds = s2s.model.predict(gen, batch_size=1, steps=1, verbose=0)
    print(preds.shape)
    tokens = np.argmax(preds, axis=2)
    print(tokens) # might be very large; look at slice instead if so
    kmers = []
    for t in tokens[0]:
        kmers.append(otokens.token(t))
    print(kmers)

