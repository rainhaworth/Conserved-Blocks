# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide TF debugging info
import model.input as dd
import model.gensynth as gs
import numpy as np
import tensorflow as tf
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lenseq', default=2000, type=int)
parser.add_argument('-s', '--shared', default=1000, type=int)
parser.add_argument('-b', '--batchsize', default=128, type=int)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-d', '--d_model', default=64, type=int)
parser.add_argument('-z', '--hashsz', default=64, type=int) # -h is help so we're using -z
parser.add_argument('-n', '--n_hash', default=8, type=int)
args = parser.parse_args()

chunksz = args.lenseq
min_len = args.shared
batch_size = args.batchsize
k = args.k
d_model = args.d_model
hashsz = args.hashsz
n_hash = args.n_hash # for bucketing approach

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
#gen = gs.gen_adversarial_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
#gen = gs.gen_simple_block_data_binary(max_len=max_len, min_len=min_len, batch_size=batch_size, tokens=itokens, k=k)
#gen_train = gs.gen_simple_block_data_binary(max_len=chunksz, min_len=chunksz, block_min=min_len, batch_size=batch_size, tokens=itokens, k=k)
gen_train = gs.gen_adversarial_chunks_binary(chunk_size=chunksz, min_shared=min_len, batch_size=batch_size,
                                             tokens=itokens, k=k, boundary_pad=20, min_pop=1.0)

print('kmer dict size:', itokens.num())

from model.chunk_hash import DiscretizedChunkHash, ChunkHash, ChunkMultiHash

#ssb = ChunkMultiHash(itokens, chunksz=chunksz, d_model=d_model, hashsz=hashsz, n_hash=n_hash)

from model.graph_hash import GraphHash

#ssb = GraphHash(itokens, k, chunksz, d_model, hashsz=hashsz, n_hash=n_hash)

from model.rnn_hash import GRUHash

ssb = GRUHash(itokens, chunksz, d_model, hashsz, n_hash)

def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * np.exp(-0.5)

mfile = '/fs/nexus-scratch/rhaworth/models/chunkhash.model.h5'

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
model_saver = tf.keras.callbacks.ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

# tensorboard
import datetime
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

ssb.compile()
try: ssb.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
    print('not implemented')
elif 'test' in sys.argv:
    print('not implemented')
else:
    ssb.model.summary()
    ssb.model.fit(gen_train, steps_per_epoch=100, epochs=15, \
                #validation_data=([Xvalid, Yvalid], None), \
                callbacks=[lr_scheduler,
                           #model_saver,
                           tensorboard_callback
                           ])
    print('done training')

    # check accuracy near decision boundary
    print('hard gen eval')
    gen_hard = gs.gen_adversarial_chunks_binary(chunk_size=chunksz, min_shared=min_len, batch_size=batch_size,
                                                tokens=itokens, k=k, boundary_pad=20, min_pop=1.0)
    ssb.model.evaluate(gen_hard, steps=100)

    # check accuracy far from decision boundary
    print('simple gen eval')
    gen_simple = gs.gen_adversarial_chunks_binary(chunk_size=chunksz, min_shared=min_len, batch_size=batch_size,
                                                  prob_sub=0.0, exp_indel_rate=0.0, exp_indel_size=0,
                                                  tokens=itokens, k=k, boundary_pad=20, min_pop=1.0)
    ssb.model.evaluate(gen_simple, steps=100)
